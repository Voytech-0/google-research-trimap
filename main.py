from data import load_data
from model import train_model, get_embeddings
import trimap
import jax.random as random

if __name__ == "__main__":
    train_dataset, val_dataset = load_data(batch_size=2)
    model, state = train_model(train_dataset, val_dataset)

    test_dataset = load_data(split='test', batch_size=2)
    embeddings, predicted, actual = get_embeddings(model, state, test_dataset)

    key = random.PRNGKey(0)
    trimap_embeddings = trimap.transform(key, embeddings, distance='euclidean')
