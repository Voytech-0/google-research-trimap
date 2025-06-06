import matplotlib.pyplot as plt

def plot_embeddings(embeddings, predicted, actual):
    label2class = ['dorsal', 'palm', 'bad']
    color_map = {'dorsal': 'blue', 'palm': 'green', 'bad': 'red'}
    fig, axes = plt.subplots(2, figsize=(10, 10))
    mistake = [int(gt != pred) for gt, pred in zip(actual, predicted)]
    print(f'accuracy:', 1 - sum(mistake) / len(mistake))
    for idx, (ax, labels) in enumerate(zip(axes, [actual, predicted])):
        colors = [color_map[label2class[label]] for label in labels]
        alpha = 1 if idx == 0 else mistake
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, marker='o', s=s,
                   alpha=alpha)

    plt.show()