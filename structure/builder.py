from collections import OrderedDict

import torch

import structure.model
from concern.config import Configurable, State


class Builder(Configurable):
    model = State()
    model_args = State()

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        if 'backbone' in cmd:
            self.model_args['backbone'] = cmd['backbone']

    @property
    def model_name(self):
        return self.model + '-' + getattr(structure.model, self.model).model_name(self.model_args)

    def build(self, device, distributed=False, local_rank: int = 0):
        Model = getattr(structure.model,self.model)
        model = Model(self.model_args, device,
                      distributed=distributed, local_rank=local_rank)
        return model

