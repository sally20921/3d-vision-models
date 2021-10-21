from typing import Optional, Union

import torch

Device = Union[str, torch.device]

def make_device(device: Device) -> torch.device:
    
