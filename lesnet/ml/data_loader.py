"""tf.data input pipeline from a manifest (paper §5.2, §5.3).

Metadata and labels are precomputed as arrays; images are loaded and preprocessed lazily
via the numpy PreprocessingPipeline wrapped in tf.py_function. Only records with a
mappable triage label are kept.
"""
import numpy as np
import tensorflow as tf

from lesnet.ml.features import metadata_vector
from lesnet.ml.preprocessing import PreprocessingPipeline
from lesnet.ml.taxonomy import fine_index, triage_index


def filter_valid(records):
    return [record for record in records if triage_index(record.raw_label) is not None]


def make_dataset(records, config, fine_vocabulary, training=False):
    records = filter_valid(records)
    if not records:
        raise ValueError("No records with a mappable triage label.")

    paths = [record.image_path for record in records]
    metadata = np.stack([metadata_vector(record) for record in records]).astype('float32')
    triage = np.array([triage_index(record.raw_label) for record in records], dtype='int64')
    n_fine = max(len(fine_vocabulary), 1)

    def _fine(record):
        index = fine_index(record.raw_label, fine_vocabulary)
        return index if index is not None else 0

    fine = np.array([_fine(record) for record in records], dtype='int64')

    pipeline = PreprocessingPipeline(image_size=config.image_size, remove_hair=not config.smoke)

    def _load_image(path_tensor):
        from PIL import Image
        image = Image.open(path_tensor.numpy().decode('utf-8')).convert('RGB')
        return pipeline(np.asarray(image)).astype('float32')

    def _map(path, meta, triage_label, fine_label):
        image = tf.py_function(_load_image, [path], tf.float32)
        image.set_shape((config.image_size[0], config.image_size[1], 3))
        return (
            {'image': image, 'metadata': meta},
            {'triage': tf.one_hot(triage_label, 3), 'fine': tf.one_hot(fine_label, n_fine)},
        )

    dataset = tf.data.Dataset.from_tensor_slices((paths, metadata, triage, fine))
    dataset = dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(paths), 1000), seed=config.seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, n_fine, triage
