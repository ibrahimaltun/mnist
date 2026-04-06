"""
If both commands return True and your GPU name is displayed, GPU support is successfully enabled.
"""

import torch
import tensorflow as tf

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 0 corresponds to the first GPU


def set_gpu_to_model():
    """Set GPU device to use"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=8192
            )]
        )
        print("It will use 8GB VRAM to train the model")
