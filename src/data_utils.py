import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def split_images_on_disc(images, disc_logits):
    if len(disc_logits.shape) == 2:
        disc_logits = tf.squeeze(disc_logits, 1)
    are_real = disc_logits >= 0.5
    return images[are_real], images[~are_real]


def disc_preds_to_label(disc_logits):
    if len(disc_logits.shape) == 2:
        disc_logits = tf.squeeze(disc_logits, 1)
    disc_p = tf.math.sigmoid(disc_logits)
    labels = np.where((disc_p >= 0.5).numpy(), "Real", "False")
    return labels


def decode_img(img, img_height, img_width):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_width, img_height])


def process_path(file_path, img_height, img_width):
    # label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_height, img_width)
    return img


def show_batch(image_batch, labels=None):
    if (image_batch < 0).numpy().any():
        image_batch = (image_batch + 1) / 2
    fig = plt.figure(figsize=(10, 10))
    for n in range(min(25, image_batch.shape[0])):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        try:
            plt.title(labels[n])
        except:
            pass
        plt.axis("off")
    return fig


def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    # ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=batch_size)

    return ds


def image_grid(x, size=5):
    t = tf.unstack(x[: size * size], num=size * size, axis=0)
    rows = [tf.concat(t[i * size : (i + 1) * size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]
