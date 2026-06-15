"""Post-training quantisation + TFLite export for the live-demo (M4.5s) model.

int8 with a representative dataset shrinks the model ~4x and lets the demo serve via a
lightweight TFLite runtime under the 500 MB peak-memory budget, while the float students/
teacher stay available for maximum accuracy. float16 is a safe fallback if a layer refuses
int8. Distillation keeps the quantised model's accuracy close to the teacher's.
"""
import os
import resource

import numpy as np
import tensorflow as tf


def representative_dataset(dataset, num_batches=64):
    """Yield model inputs (image, metadata) for the int8 calibration pass."""
    def generator():
        for count, (inputs, _targets) in enumerate(dataset):
            if count >= num_batches:
                break
            yield [tf.cast(inputs['image'], tf.float32), tf.cast(inputs['metadata'], tf.float32)]
    return generator


def export_tflite(model, path, dataset=None, mode='int8', num_batches=64):
    """Export ``model`` to a TFLite flatbuffer. mode='int8' (needs dataset) or 'float16'."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == 'int8':
        if dataset is None:
            raise ValueError("int8 quantisation needs a representative dataset.")
        converter.representative_dataset = representative_dataset(dataset, num_batches)
    elif mode == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    else:
        raise ValueError(f"Unknown quantisation mode '{mode}'.")
    flatbuffer = converter.convert()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(flatbuffer)
    return path


def model_size_mb(path):
    return os.path.getsize(path) / 1e6


def peak_rss_mb():
    """Process peak resident memory in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tflite_triage_logits(tflite_path, image, metadata):
    """Run the quantised model and return triage logits (live-demo inference path)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    by_pixels = sorted(inputs, key=lambda detail: int(np.prod(detail['shape'])), reverse=True)
    interpreter.set_tensor(by_pixels[0]['index'], image.astype(by_pixels[0]['dtype']))
    interpreter.set_tensor(by_pixels[1]['index'], metadata.astype(by_pixels[1]['dtype']))
    interpreter.invoke()
    outputs = interpreter.get_output_details()
    triage = min(outputs, key=lambda detail: int(np.prod(detail['shape'])))  # 3-way head is smallest
    return interpreter.get_tensor(triage['index'])
