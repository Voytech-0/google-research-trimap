import os
import tensorflow as tf
import tensorflow_datasets as tfds

import kagglehub


# def numpy_collate(batch):
#     return tree_map(np.asarray, default_collate(batch))

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=7)
    return image, label

def prepare(dataset):
    dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return dataset


def load_data(domain="art_painting",  split='train', batch_size=32, image_size=224, val_split=0.2, seed=0):
    # Download and locate PACS dataset
    dataset_path = kagglehub.dataset_download("nickfratto/pacs-dataset")
    data_dir = os.path.join(dataset_path, "pacs_data", "pacs_data", domain)

    # Use TensorFlow's image_dataset_from_directory
    dataloader_args = {
        'image_size': image_size,
        'batch_size': batch_size,
        'label_mode': 'int',
        'seed': seed,
        'validation_split': val_split,
    }

    if split == 'train':
        train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, **dataloader_args,
                                                                    subset='training', shuffle=True)
        val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, **dataloader_args,
                                                                  subset='validation', shuffle=False)

        val_dataset = prepare(val_dataset)
        train_dataset = prepare(train_dataset)
        return train_dataset, val_dataset
    else:
        dataset = tf.keras.utils.image_dataset_from_directory(data_dir, **dataloader_args, shuffle=False)
        dataset = prepare(dataset)
        return dataset
