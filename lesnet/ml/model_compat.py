"""Compatibility layer for loading legacy Keras models with deprecated parameters.

Handles deserialization of older models that use deprecated BatchNormalization
parameters (renorm, renorm_clipping, renorm_momentum) which are no longer
supported in Keras 3.x.
"""
import json
from pathlib import Path

import tensorflow as tf


def _filter_deprecated_bn_params(config):
    """Recursively filter deprecated BatchNormalization parameters from model config.
    
    Removes renorm, renorm_clipping, and renorm_momentum which are not supported
    in Keras 3.x BatchNormalization layers.
    """
    if isinstance(config, dict):
        # Check if this is a BatchNormalization layer config
        if config.get('class_name') == 'BatchNormalization':
            # Remove deprecated parameters
            deprecated_params = {'renorm', 'renorm_clipping', 'renorm_momentum'}
            layer_config = config.get('config', {})
            for param in deprecated_params:
                layer_config.pop(param, None)
        
        # Recursively process nested configs
        for key, value in config.items():
            config[key] = _filter_deprecated_bn_params(value)
    
    elif isinstance(config, list):
        return [_filter_deprecated_bn_params(item) for item in config]
    
    return config


def load_model_with_compatibility(model_path, **kwargs):
    """Load a Keras model with automatic compatibility fixes for deprecated parameters.
    
    This function handles loading legacy models that may contain deprecated
    BatchNormalization parameters by filtering them out before deserialization.
    
    Args:
        model_path: Path to the model file (.keras or .h5 format)
        **kwargs: Additional arguments to pass to tf.keras.models.load_model
    
    Returns:
        Loaded Keras model
        
    Raises:
        ValueError: If the model file cannot be read or deserialized even after fixes
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # For .keras format files, we need to extract and fix the config
    if model_path.suffix == '.keras':
        import zipfile
        import tempfile
        import os
        
        try:
            # Extract the keras model (which is a zip file)
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Load and fix the model config
                config_path = Path(temp_dir) / 'model.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Filter deprecated parameters
                    fixed_config = _filter_deprecated_bn_params(config)
                    
                    # Write back the fixed config
                    with open(config_path, 'w') as f:
                        json.dump(fixed_config, f)
                
                # Create a temporary fixed model file
                fixed_model_path = Path(temp_dir) / 'model_fixed.keras'
                
                # Re-zip with fixed config
                with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file != 'model_fixed.keras':
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(temp_dir)
                                zipf.write(file_path, arcname)
                
                # Load the fixed model
                return tf.keras.models.load_model(str(fixed_model_path), **kwargs)
        
        except Exception as e:
            # If the fix doesn't work, try loading normally (might work with newer Keras)
            try:
                return tf.keras.models.load_model(str(model_path), **kwargs)
            except Exception:
                raise ValueError(
                    f"Failed to load model from {model_path}. "
                    f"The model may be incompatible with this version of Keras. "
                    f"Original error: {e}"
                )
    
    else:
        # For other formats (.h5, etc.), use standard loading
        return tf.keras.models.load_model(str(model_path), **kwargs)
