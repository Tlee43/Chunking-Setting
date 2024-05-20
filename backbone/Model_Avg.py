import torch
from torch import nn
from copy import deepcopy


# modified from timm library
class ModelAvg(nn.Module):
    def __init__(self, model, avg_func, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.avg_func = avg_func
        self.device = device  # perform ema on different device from model if set
        self.num_of_updates = 1
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
        self.num_of_updates += 1

    def update(self, model):
        self._update(model, update_fn=self.avg_func)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMA(ModelAvg):
    def __init__(self, model, decay=0.95, device=None):
        self.decay = decay
        super(EMA, self).__init__(model, lambda e, m: self.decay * e + (1. - self.decay) * m, device)


class AvgOverChunks(ModelAvg):
    def __init__(self, model, device=None):
        super(AvgOverChunks, self).__init__(model, lambda e, m: (self.num_of_updates * e + m)/(self.num_of_updates + 1), device)
