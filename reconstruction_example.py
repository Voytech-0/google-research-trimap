import jax.random as random
from sklearn.datasets import load_digits
import trimap.trimap as trimap
import trimap.parametric_trimap as ptrimap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from visualize import plot_embeddings, plot_inverse_grid

def generate_grid(top_left, bottom_right, num_x=10, num_y=10):
    # Define corners
    top_left = np.array(top_left)
    bottom_right = np.array(bottom_right)
    top_right = np.array([bottom_right[0], top_left[1]])
    bottom_left = np.array([top_left[0], bottom_right[1]])

    # Create grid
    xs = np.linspace(0, 1, num_x)
    ys = np.linspace(0, 1, num_y)
    grid = []

    for y in ys:
        for x in xs:
            point = (
                    (1 - x) * (1 - y) * top_left +
                    x * (1 - y) * top_right +
                    (1 - x) * y * bottom_left +
                    x * y * bottom_right
            )
            grid.append(point)

    return np.array(grid)


def parametric_trimap(key, data):
    embedding, model, params = ptrimap.fit_transform(key, data, 2, verbose=True)
    test_pts = generate_grid(np.min(embedding, axis=0), np.max(embedding, axis=0))
    reconstructed_data = ptrimap.inverse_transform(test_pts, model, params)
    return embedding, reconstructed_data, test_pts

def iterative_trimap(key, data):
    forward_transform_key, inverse_transform_key = random.split(key, 2)
    embedding = trimap.transform(forward_transform_key, data, 2, verbose=True)
    test_pts = generate_grid(np.min(embedding, axis=0), np.max(embedding, axis=0))
    reconstructed_data = trimap.inverse_transform(inverse_transform_key, test_pts, embedding, data, verbose=True, n_iters=20)
    return embedding, reconstructed_data, test_pts

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='mnist')
    args.add_argument('--parametric', action='store_true', default=False)
    args = args.parse_args()
    data = load_digits()
    digits = data.data
    key = random.PRNGKey(42)
    original_shape = digits.shape[0], 8, 8

    # Configure logging to show INFO level messages and above
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
    )

    trimap_fn = parametric_trimap if args.parametric else iterative_trimap

    embedding, reconstructed_digits, test_pts = trimap_fn(key, digits)
    plot_embeddings(embedding, data.target)

    reconstructed_digits = reconstructed_digits.reshape((-1, *original_shape[1:]))
    digits = digits.reshape(original_shape)
    print(f'shapes: embedding {embedding.shape} reconstructed {reconstructed_digits.shape}')
    plot_inverse_grid(embedding, reconstructed_digits, data.target, test_pts)