import numpy
import torch.nn as nn

class LossBase(nn.Module):
    '''
    base class for losses.
    '''
    def __init__(self):
        ''' 
        initializes logs and metrics dictionaries
        '''
        super().__init__()
        self._logs = {}
        self._metrics = {}

    @property
    def logs(self):
        '''
        return logs
        '''
        return self._logs

    @property
    def metrics(self):
        '''
        returns metrics
        '''
        return self._metrics

    def add_metric(self, key, val):
        '''
        add a new metric to the dictionary and detach it
        '''
        self._metrics[key] = val.detach()

