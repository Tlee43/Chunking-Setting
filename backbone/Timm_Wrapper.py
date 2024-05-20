import timm
from backbone import MammothBackbone
import torch.nn as nn
import torch

class TimmModel(MammothBackbone):

    def __init__(self, model_name, pretrained, num_classes, linear_probe=False):
        super(MammothBackbone, self).__init__()
        self.linear_probe = linear_probe
        if linear_probe:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            self.model.eval()
            for param in self.model.parameters():
                param.grad = None
                param.requires_grad = False
            self.head = nn.Linear(self.model.fc.weight.shape[1], num_classes)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def _lin_forward(self, x, returnt='out'):
        with torch.no_grad():
            x = self.model.forward_features(x)
            features = self.model.forward_head(x, pre_logits=True)
        if returnt == 'features':
            return features
        out = self.head(features)
        if returnt == 'out':
            return out
        elif returnt == 'all':
            # this might not always work :(
            return (out, features)

        raise NotImplementedError("Unknown return type")

    
    def _full_forward(self, x, returnt='out'):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """
        x = self.model.forward_features(x)

        if returnt == 'features':
            return self.model.forward_head(x, pre_logits=True)

        if returnt == 'out':
            return self.model.forward_head(x)
        elif returnt == 'all':
            # this might not always work :(
            features = self.model.forward_head(x, pre_logits=True)
            out = self.model.fc(features)
            return (out, features)

        raise NotImplementedError("Unknown return type")

    def forward(self, x, returnt='out'):
        if self.linear_probe:
            return self._lin_forward(x, returnt)
        else:
            return self._full_forward(x, returnt)    


