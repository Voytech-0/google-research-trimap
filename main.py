from argparse import ArgumentParser

from data import load_data
from model import train_model, get_embeddings, LightningModel
from trimap import trimap
import jax.random as random

from visualize import plot_embeddings


def test_embeddings():
    model = LightningModel.load_from_checkpoint('best.ckpt').model
    model.eval()  # important for inference
    test_dataset = load_data(split='test', batch_size=32)
    embeddings, predicted, actual = get_embeddings(model, test_dataset)
    key = random.PRNGKey(42)

    embeddings = trimap.transform(key, embeddings, distance='euclidean')
    plot_embeddings(embeddings, predicted, actual)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--test', action='store_true', default=False)
    args.add_argument('--train', action='store_true', default=False)

    args = args.parse_args()
    if args.train:
        train_dataset, val_dataset = load_data(batch_size=32)
        train_model(train_dataset, val_dataset)

    if args.test:
        test_embeddings()






