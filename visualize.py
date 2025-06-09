import matplotlib.pyplot as plt

def plot_embeddings(embeddings, actual, predicted=None):
    if predicted is None:
        predicted = actual
    # color_map = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'pink', 'orange', 'turquoise']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    mistake = [int(gt != pred) for gt, pred in zip(actual, predicted)]
    s = 50
    print(f'accuracy:', 1 - sum(mistake) / len(mistake))
    for idx, (ax, labels) in enumerate(zip(axes, [actual, predicted])):
        alpha = 1 if idx == 0 else mistake
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, marker='o', s=s,
                   alpha=alpha)

    plt.show()