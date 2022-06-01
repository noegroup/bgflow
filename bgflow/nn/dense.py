import torch

from ..utils.types import is_list_or_tuple


__all__ = ["DenseNet", "MeanFreeDenseNet"]


class DenseNet(torch.nn.Module):
    def __init__(self, n_units, activation=None, weight_scale=1.0, bias_scale=0.0):
        """
            Simple multi-layer perceptron.

            Parameters:
            -----------
            n_units : List / Tuple of integers.
            activation : Non-linearity or List / Tuple of non-linearities.
                If List / Tuple then each nonlinearity will be placed after each respective hidden layer.
                If just a single non-linearity, will be applied to all hidden layers.
                If set to None no non-linearity will be applied.
        """
        super().__init__()

        dims_in = n_units[:-1]
        dims_out = n_units[1:]

        if is_list_or_tuple(activation):
            assert len(activation) == len(n_units) - 2

        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(dims_in, dims_out)):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers[-1].weight.data *= weight_scale
            if bias_scale > 0.0:
                layers[-1].bias.data = (
                    torch.Tensor(layers[-1].bias.data).uniform_() * bias_scale
                )
            if i < len(n_units) - 2:
                if activation is not None:
                    if is_list_or_tuple(activation):
                        layers.append(activation[i])
                    else:
                        layers.append(activation)

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class MeanFreeDenseNet(DenseNet):
    def forward(self, x):
        y = self._layers(x)
        return y - y.mean(dim=1, keepdim=True)
