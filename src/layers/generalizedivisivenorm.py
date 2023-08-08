import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F
from torch.nn import Parameter
from ..util.math import lower_bound

class Parameterizer(object):

    def __init__(self):
        pass

    def init(self, param):
        raise NotImplementedError()

    def __call__(self, param):
        raise NotImplementedError()


class NonnegativeParameterizer(Parameterizer):
    """Object encapsulating nonnegative parameterization as needed for GDN.

    The variable is subjected to an invertible transformation that slows down the
    learning rate for small values.

    Args:
        offset: Offset added to the reparameterization of beta and gamma.
            The reparameterization of beta and gamma as their square roots lets the
            training slow down when their values are close to zero, which is desirable
            as small values in the denominator can lead to a situation where gradient
            noise on beta/gamma leads to extreme amounts of noise in the GDN
            activations. However, without the offset, we would get zero gradients if
            any elements of beta or gamma were exactly zero, and thus the training
            could get stuck. To prevent this, we add this small constant. The default
            value was empirically determined as a good starting point. Making it
            bigger potentially leads to more gradient noise on the activations, making
            it too small may lead to numerical precision issues.
    """

    def __init__(self, offset=2 ** -18):
        self.pedestal = offset ** 2

    def init(self, param):
        """no grad init data"""
        with torch.no_grad():
            data = param.relu().add(self.pedestal).sqrt()

        return Parameter(data)

    def __call__(self, param, minmum=0):
        """reparam data"""
        bound = (minmum + self.pedestal) ** 0.5
        return lower_bound(param, bound).pow(2) - self.pedestal



def generalized_divisive_norm(input, gamma, beta, inverse: bool, simplify: bool = True, eps: float = 1e-5):
    """generalized divisive normalization"""
    C1, C2 = gamma.size()
    assert C1 == C2, "gamma must be a square matrix"

    x = input.view(input.size()[:2] + (-1,))
    gamma = gamma.reshape(C1, C2, 1)

    # Norm pool calc
    if simplify:
        norm_pool = F.conv1d(x.abs(), gamma, beta.add(eps))
    else:
        norm_pool = F.conv1d(x.pow(2), gamma, beta.add(eps)).sqrt()

    # Apply norm
    if inverse:
        output = x * norm_pool
    else:
        output = x / norm_pool

    return output.view_as(input)


class GeneralizedDivisiveNorm(Module):
    """Generalized divisive normalization layer.

    .. math::
        y[i] = x[i] / sqrt(sum_j(gamma[j, i] * x[j]^2) + beta[i])
        if simplify
        y[i] = x[i] / (sum_j(gamma[j, i] * |x[j]|) + beta[i])

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        inverse: If `False`, compute GDN response. If `True`, compute IGDN
            response (one step of fixed point iteration to invert GDN; the division is
            replaced by multiplication). Default: False.
        gamma_init: The gamma matrix will be initialized as the identity matrix
            multiplied with this value. If set to zero, the layer is effectively
            initialized to the identity operation, since beta is initialized as one. A
            good default setting is somewhere between 0 and 0.5.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.

    Shape:
        - Input: :math:`(B, C)`, `(B, C, L)`, `(B, C, H, W)` or `(B, C, D, H, W)`
        - Output: same as input

    Reference:
        paper: https://arxiv.org/abs/1511.06281
        github: https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py
    """

    def __init__(self, num_features, inverse: bool = False, simplify: bool = True, gamma_init: float = .1, eps: float = 1e-5):
        super(GeneralizedDivisiveNorm, self).__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.simplify = simplify
        self.gamma_init = gamma_init
        self.eps = eps

        self.weight = Parameter(torch.Tensor(num_features, num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.parameterizer = NonnegativeParameterizer()

        self.reset_parameters()

    def reset_parameters(self):
        weight_init = torch.eye(self.num_features) * self.gamma_init
        self.weight = self.parameterizer.init(weight_init)
        self.bias = self.parameterizer.init(torch.ones(self.num_features))

    def extra_repr(self):
        s = '{num_features}'
        if self.inverse:
            s += ', inverse=True'
        if self.simplify:
            s += ', simplify=True'
        s += ', gamma_init={gamma_init}, eps={eps}'
        return s.format(**self.__dict__)

    @property
    def gamma(self):
        return self.parameterizer(self.weight)

    @property
    def beta(self):
        return self.parameterizer(self.bias)

    def forward(self, input):
        return generalized_divisive_norm(input, self.gamma, self.beta, self.inverse, self.simplify, self.eps)
