import jax.random as random
from sklearn.datasets import load_digits
import trimap.trimap as trimap
import trimap.parametric_trimap as ptrimap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import logging

from visualize import plot_embeddings

data = load_digits()
digits = data.data
key = random.PRNGKey(42)
original_shape = digits.shape[0], 8, 8

# Configure logging to show INFO level messages and above
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
)
embedding, model, params = ptrimap.fit_transform(digits, 2, key, verbose=True)
plot_embeddings(embedding, data.target)

reconstructed_digits = ptrimap.inverse_transform(embedding, model, params)
reconstructed_digits = reconstructed_digits.reshape(original_shape)
digits = digits.reshape(original_shape)
fig, axs = plt.subplots(3, 2)
for i in range(3):
    axs[i][0].imshow(digits[i], cmap='gray')
    axs[i][1].imshow(reconstructed_digits[i], cmap='gray')

plt.show()