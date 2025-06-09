from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from trimap.trimap import generate_triplets, trimap_loss

from absl import logging

class ParametricTriMap(nn.Module):
    input_dims: int
    latent_dims: int
    hidden_dims: int = 100
    hidden_layers: int = 3
    activation_fn: callable = nn.relu
    kernel_init: callable = nn.initializers.kaiming_normal()
    bias_init: callable = nn.initializers.zeros

    def setup(self):
        forwarded_params = (self.hidden_dims, self.hidden_layers, self.activation_fn, self.kernel_init, self.bias_init)
        self.encoder = Encoder(self.latent_dims, *forwarded_params)
        self.decoder = Decoder(self.input_dims, *forwarded_params)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

    @nn.compact
    def __call__(self, x):
        approx_trimap_embedding = self.encode(x)
        approx_x = self.decode(approx_trimap_embedding)
        return approx_x

class Encoder(nn.Module):
    latent_dims: int
    hidden_dims: int
    hidden_layers: int
    activation_fn: callable
    kernel_init: callable
    bias_init: callable

    @nn.compact
    def __call__(self, x):
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation_fn(nn.Dense(self.hidden_dims)(x))
        latent = nn.Dense(self.latent_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return latent

class Decoder(nn.Module):
    out_dims: int
    hidden_dims: int
    hidden_layers: int
    activation_fn: callable
    kernel_init: callable
    bias_init: callable

    @nn.compact
    def __call__(self, x):
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activation_fn(x)
        output = nn.Dense(self.out_dims, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return output

def initialize_model(input_dims, n_dims, rng_key, lr=1e-4):
    autoencoder = ParametricTriMap(input_dims, n_dims)

    dummy_input = jnp.ones((1, input_dims))
    variables = autoencoder.init(rng_key, dummy_input)
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=autoencoder.apply, params=variables['params'], tx=tx)
    return autoencoder, state


def parametric_triplet_loss(params, state, embedding, triplets, weights, alpha=0.3):
    compressed_embedding = state.apply_fn({'params': params}, embedding, method=ParametricTriMap.encode)
    triplet_loss = trimap_loss(compressed_embedding, triplets, weights)
    reconstruction = state.apply_fn({'params': params}, compressed_embedding, method=ParametricTriMap.decode)
    reconstruction_loss = jnp.mean(optax.huber_loss(reconstruction, embedding))
    loss = (1 - alpha) * triplet_loss + alpha * reconstruction_loss
    return loss, {'triplet_loss': triplet_loss, 'reconstruction_loss': reconstruction_loss}


@partial(jax.jit, static_argnames=['alpha'])
def train_step(state, embedding, triplets, weights, alpha):
    grad_fn = jax.value_and_grad(parametric_triplet_loss, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, state, embedding, triplets, weights, alpha)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux


def fit(inputs, n_dims, rng_key,
                  lr=1e-4,
                  n_inliers=10,
                  n_outliers=5,
                  n_random=3,
                  batch_size=32,
                  n_epochs=1000,
                  reconstruction_loss_weight=0.05,
                  weight_temp=0.5,
                  distance='euclidean',
                  verbose=False):
    model, state = initialize_model(inputs.shape[1], n_dims, rng_key, lr)
    triplets, weights = generate_triplets(
        rng_key,
        inputs,
        n_inliers,
        n_outliers,
        n_random,
        weight_temp=weight_temp,
        distance=distance,
        verbose=verbose)

    inputs = jnp.asarray(inputs)
    for epoch in range(n_epochs):
        state, loss, aux = train_step(state, inputs, triplets, weight_temp, reconstruction_loss_weight)
        if verbose:
            logging.info(f'Epoch {epoch} loss: {loss:.3}, '
                         f'trimap_loss {aux["triplet_loss"]:.3}, reconstruction_loss {aux["reconstruction_loss"]:.3}')


    return model, state.params

def transform(inputs, model, params):
    return model.apply({'params': params}, inputs, method=ParametricTriMap.encode)

def fit_transform(inputs, n_dims, rng_key,
                  lr=1e-4,
                  n_inliers=10,
                  n_outliers=5,
                  n_random=3,
                  batch_size=32,
                  n_epochs=1000,
                  reconstruction_loss_weight=0.05,
                  weight_temp=0.5,
                  distance='euclidean',
                  verbose=False):
    model, params = fit(inputs, n_dims, rng_key, lr, n_inliers, n_outliers, n_random,
                        batch_size=batch_size, n_epochs=n_epochs, reconstruction_loss_weight=reconstruction_loss_weight,
                        weight_temp=weight_temp, distance=distance, verbose=verbose)

    embedding = transform(inputs, model, params)
    return embedding, model, params

def inverse_transform(embedding, model, params):
    return model.apply({'params': params}, embedding, method=ParametricTriMap.decode)
