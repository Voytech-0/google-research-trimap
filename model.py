import flaxmodels as fm
import jax
import jax.numpy as jnp
import optax
from flax.linen import tabulate
from flax.training import train_state
from flax import linen as nn
import orbax.checkpoint as ocp

import jax.random as random

class TransferredPretrained(nn.Module):
    base_model: nn.Module
    final_activation: str
    num_classes: int

    @nn.compact
    def __call__(self, x, train=False, get_embedding=False): # train=False means no dropout
        x = self.base_model(x, train=train)[self.final_activation]
        x = jnp.mean(x, axis=(1, 2)) # average pooling
        if get_embedding:
            return x
        x = nn.Dense(self.num_classes, name="classifier")(x)
        return x

def load_pretrained_backbone(backbone='resnet50'):
    if backbone == 'resnet50':
        return fm.ResNet50(pretrained='imagenet', output='activations'), 'block4_2'
    elif backbone == 'resnet18':
        return fm.ResNet18(pretrained='imagenet', output='activations'), 'block4_1'
    else:
        raise ValueError(f'{backbone} is an unknown backbone')

def visualize_model():
    base_model, final_activation = load_pretrained_backbone('resnet18')
    model = TransferredPretrained(base_model, final_activation, 7)

    dummy_input = jnp.ones((1, 224, 224, 3))
    print(tabulate(model, random.PRNGKey(0))(dummy_input, train=False))
    exit(0)

def load_model(rng_key, num_classes=7):
    base_model, final_activation = load_pretrained_backbone('resnet18')
    model = TransferredPretrained(base_model, final_activation, 7)

    dummy_input = jnp.ones((1, 224, 224, 3))
    variables = model.init(rng_key, dummy_input, train=False)

    return model, variables['params'], variables['batch_stats']

def compute_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

@jax.jit
def train_step(state, x, y, batch_stats):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats}, x, train=True, mutable=['batch_stats'])
        loss = compute_loss(logits, y)
        return loss, updates

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, updates['batch_stats']

@jax.jit
def eval_step(state, x, y, batch_stats):
    logits = state.apply_fn({'params': state.params, 'batch_stats': batch_stats}, x)
    loss = compute_loss(logits, y)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
    return loss, accuracy

def eval(state, dataloader, batch_stats):
    loss, acc, sum_of_weights = 0, 0, 0
    batch_size = dataloader.batch_size
    for x, y in dataloader:
        weight = x.shape[0] / batch_size
        batch_loss, batch_acc = eval_step(state, x, y, batch_stats)
        loss += batch_loss * weight
        acc += batch_acc * weight
        sum_of_weights += weight
    loss /= sum_of_weights
    acc /= sum_of_weights
    return loss, acc

def create_train_state(model, params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def setup_checkpoints():
    options = ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)

    # Specify the directory to save checkpoints
    checkpoint_dir = "/tmp/my_model_checkpoints"

    # Create the CheckpointManager
    ckpt_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    )
    return ckpt_manager


def train_model(train_dataloader, eval_dataloader, seed=0):
    rng = random.PRNGKey(0)
    model, params, batch_stats = load_model(rng, num_classes=7)
    state = create_train_state(model, params, learning_rate=1e-3)
    best_val_accuracy = 0
    ckpt_manager = setup_checkpoints()
    for epoch in range(10):
        for x, y in train_dataloader:
            state, batch_stats = train_step(state, x, y, batch_stats)

        train_loss, train_acc = eval(state, train_dataloader, batch_stats)
        val_loss, val_acc = eval(state, eval_dataloader, batch_stats)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},"
              f" Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_args = ocp.args.Composite(
                train_state=ocp.args.StandardSave(state),  # Save the TrainState
                batch_stats=ocp.args.StandardSave(batch_stats)  # Save the batch_stats PyTree
            )
            ckpt_manager.save(epoch, args=save_args)

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()
    return model, state

def get_embeddings(model, state, dataloader):
    predictor_head = nn.Dense(model.num_classes)
    features = []
    predictions = []
    actual = []
    for x, y in dataloader:
        batch_features = model.base_model.apply({'params': state.params['base_model'], 'batch_stats': state.batch_stats},
                                                x, train=False, get_embedding=True)
        features.append(batch_features)
        predictions.append(predictor_head.apply({'params': state.params['classifier']}, batch_features))
        actual.append(y)

    return features, predictions, actual


