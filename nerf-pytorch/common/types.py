from typing import Optional, Union

import torch

Device = Union[str, torch.device]

def make_device(device: Device) -> torch.device:
    '''
    Makes an actual torch.device object from the device
    specified as either a string or torch.device object.
    If the device is `cuda` without a specific index, 
    the index of the current device is assigned.

    Args
    ____
        device: Device (as str or torch.device)
    
    Returns
    _____
        a matching torch.device object 
    '''

    
