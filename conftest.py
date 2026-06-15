"""Pytest session setup.

Tests run on CPU: they must be deterministic and GPU-independent (CI has no GPU, and
the pip TensorFlow wheel has no bundled CUDA libdevice for XLA). Real GPU training runs
outside the test suite. Set before TensorFlow initialises its devices.
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
