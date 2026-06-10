"""TFLite export (paper §7) — a working replacement for the broken quantize path."""
import tensorflow as tf


def export_tflite(model, path, quantize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    flatbuffer = converter.convert()
    with open(path, 'wb') as handle:
        handle.write(flatbuffer)
    return path
