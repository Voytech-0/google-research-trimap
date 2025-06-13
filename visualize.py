import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_embeddings(embeddings, actual, predicted=None):
    if predicted is None:
        predicted = actual
    # color_map = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'pink', 'orange', 'turquoise']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    mistake = [int(gt != pred) for gt, pred in zip(actual, predicted)]
    print(f'accuracy:', 1 - sum(mistake) / len(mistake))
    for idx, (ax, labels) in enumerate(zip(axes, [actual, predicted])):
        alpha = 1 if idx == 0 else mistake
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, marker='o', s=50,
                   alpha=alpha, cmap='tab10')

    plt.show()

def plot_inverse_grid(embeddings, inverse_transformed_embeddings, labels, test_points):
    # Set up the grid
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(10, 20, fig)
    scatter_ax = fig.add_subplot(gs[:, :10])
    digit_axes = np.zeros((10, 10), dtype=object)
    for i in range(10):
        for j in range(10):
            digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

    # Use umap.plot to plot to the major axis
    # umap.plot.points(mapper, labels=labels, ax=scatter_ax)
    scatter_ax.scatter(embeddings[:, 0], embeddings[:, 1],
                       c=labels, cmap='tab10', s=50)
    scatter_ax.set(xticks=[], yticks=[])

    # Plot the locations of the text points
    scatter_ax.scatter(test_points[:, 0], test_points[:, 1], marker='x', c='k', s=15)

    inverse_transformed_embeddings = inverse_transformed_embeddings.reshape((10, 10, *inverse_transformed_embeddings.shape[1:]))
    # Plot each of the generated digit images
    for i in range(10):
        for j in range(10):
            # axis flip bottom left->0 0 to top left -> 0, 0
            digit_axes[i, j].imshow(inverse_transformed_embeddings[-(i+1), j])
            digit_axes[i, j].set(xticks=[], yticks=[])
    plt.show()

def plot_new_insertions(embeddings, new_embeddings, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=10, cmap='tab20', alpha=0.6, label='Original Data')
    plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1],
                color='red', s=80, edgecolor='black', label='New Points')
    plt.title("TriMap Projection of CIFAR-100 with New Points Added")
    plt.legend()
    plt.tight_layout()
    plt.show()

