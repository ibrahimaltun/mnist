"""
If both commands return True and your GPU name is displayed, GPU support is successfully enabled.
"""

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 0 corresponds to the first GPU
