import tensorflow as tf
import matplotlib.pyplot as plt


def decode_img(img, img_width, img_height):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_width, img_height])


def process_path(file_path):
    # label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def show_batch(image_batch, labels=None):
    plt.figure(figsize=(10, 10))
    for n in range(min(25, image_batch.shape[0])):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        try:
            plt.title(labels[n])
        except:
            pass
        plt.axis("off")


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
