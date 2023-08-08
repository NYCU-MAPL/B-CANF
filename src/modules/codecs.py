import torch
from torch import nn
from inspect import signature
from functools import partial
from src.layers.context_model import ContextModel
from src.layers.entropy_models import __CONDITIONS__, EntropyBottleneck
from src.layers.generalizedivisivenorm import GeneralizedDivisiveNorm
from src.util.tools import Conv2d, ConvTranspose2d


############################################## for Hyperprior ###########################################
class FactorizedCoder(nn.Module):
    """FactorizedCoder"""

    def __init__(self, num_priors, quant_mode='noise'):
        super(FactorizedCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.entropy_bottleneck = EntropyBottleneck(
            num_priors, quant_mode=quant_mode)

        self.divisor = 16
        self.num_bitstreams = 1

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        ret = self.entropy_bottleneck.compress(features, return_sym=return_hat)

        if return_hat:
            y_hat, strings, shape = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, strings, shape
        else:
            return ret

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings, shape)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        y_tilde, likelihoods = self.entropy_bottleneck(features)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, likelihoods


class HyperPriorCoder(FactorizedCoder):
    """HyperPrior Coder"""

    def __init__(self, num_condition, num_priors, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(HyperPriorCoder, self).__init__(
            num_priors, quant_mode=quant_mode)
        self.use_mean = use_mean

        self.use_abs = not self.use_mean or use_abs
        self.conditional_bottleneck = __CONDITIONS__[condition](use_mean=use_mean, quant_mode=quant_mode)
        if use_context:
            self.conditional_bottleneck = ContextModel(
                num_condition, num_condition*2, self.conditional_bottleneck)
        self.hyper_analysis = nn.Sequential()
        self.hyper_synthesis = nn.Sequential()

        self.divisor = 64
        self.num_bitstreams = 2

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream, side_stream], [hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [hyperpriors.size()]

    def decompress(self, strings, shape):
        stream, side_stream = strings
        z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, condition.size(), condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, z_likelihood)


class GoogleAnalysisTransform(nn.Sequential):

    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class GoogleSynthesisTransform(nn.Sequential):

    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(GoogleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class GoogleHyperAnalysisTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )


class GoogleHyperScaleSynthesisTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperScaleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters, kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features, kernel_size=3, stride=1, parameterizer=None)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features, kernel_size=3, stride=1)
        )

############################################## for B-CANF ###########################################
class DQ_ResBlock(nn.Sequential):

    def __init__(self, num_filters):
        super().__init__(
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(num_filters, num_filters, 3)
        )

    def forward(self, input):
        return super().forward(input) + input


class DeQuantizationModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(DeQuantizationModule, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, 3)
        self.resblock = nn.Sequential(
            *[DQ_ResBlock(num_filters) for _ in range(num_layers)])
        self.conv2 = Conv2d(num_filters, num_filters, 3)
        self.conv3 = Conv2d(num_filters, out_channels, 3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3


class ANATransform(nn.Sequential):

    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(ANATransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2),
        )


class SYNTransform(nn.Sequential):

    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn):
        super(SYNTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        ) 


class HyperANATransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(HyperANATransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=3, stride=2)
        )


class HyperSYNTransform(nn.Sequential):

    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(HyperSYNTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features, kernel_size=3, stride=1)
        )


class BCANFCodec(HyperPriorCoder):

    def __init__(
            self, in_channels, cond_channels, num_filters, num_features, num_hyperpriors, kernel_size, 
            num_layers=1, hyper_filters=128, pred_kernel_size=5, num_predprior_filters=128, 
            pa_filters=640, num_DQ_layer=6, num_DQ_ch=64, use_mean=False, use_context=False, 
            simplify_gdn=False, condition='Gaussian', quant_mode='noise'
        ):
        super(BCANFCodec, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_hyperpriors = num_hyperpriors
        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        # analysis and synthesis pairs
        for i in range(num_layers):
            ks = kernel_size[i]
            self.add_module('analysis'+str(i), ANATransform(in_channels + cond_channels, num_features, num_filters, ks, simplify_gdn))
            self.add_module('synthesis'+str(i), SYNTransform(in_channels, num_features, num_filters, ks, simplify_gdn))


        # hyper prior
        self.hyper_analysis = HyperANATransform(num_features, hyper_filters, num_hyperpriors)
        self.hyper_synthesis = HyperSYNTransform(num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
   
        
        # temporal prior
        self.pred_prior = GoogleAnalysisTransform(cond_channels, num_features, num_predprior_filters,
                                                  pred_kernel_size, simplify_gdn=False)

        self.PA = nn.Sequential(
            nn.Conv2d(num_features * (1 + self.conditional_bottleneck.condition_size), pa_filters, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(pa_filters, pa_filters, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(pa_filters, num_features * self.conditional_bottleneck.condition_size, 1)
        )

        # quality enhancement network
        self.QE = DeQuantizationModule(in_channels, in_channels, num_DQ_ch, num_DQ_layer)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, x, z, h, cond_input, compress=False):
        for i in range(self.num_layers):
            inputs = torch.cat([x, cond_input], dim=1)
            z = z + self[f'analysis{i}'](inputs)

            if i < self.num_layers - 1:
                x = x - self[f'synthesis{i}'](z)

        pred_feat = self.pred_prior(cond_input)

        h = h + self.hyper_analysis(z.abs() if self.use_abs else z)
        if compress:
            stream_h, h_quant = self.entropy_bottleneck.compress(h, return_sym=True)
        else:
            h_quant, h_likelihood = self.entropy_bottleneck(h)
        condition = self.hyper_synthesis(h_quant)
        condition = self.PA(torch.cat([condition, pred_feat], dim=1))
   
        if compress:
            stream_z, z_quant = self.conditional_bottleneck.compress(z, condition=condition, return_sym=True)
        else:
            z_quant, z_likelihood = self.conditional_bottleneck(z, condition=condition)

        rx = self[f'synthesis{self.num_layers-1}'](z_quant)
        x = x - rx

        if compress:
            return x, z_quant, h_quant, stream_z, stream_h, rx
        else:
            return x, z_quant, h_quant, z_likelihood, h_likelihood, rx

    def decode(self, x, z, cond_input, rx=None):
        for i in range(self.num_layers-1, -1, -1):           
            if (i == self.num_layers - 1) and isinstance(rx, torch.Tensor):
                x = x + rx
            else:
                x = x + self[f'synthesis{i}'](z)

            if i:
                inputs = torch.cat([x, cond_input], dim=1)
                z = z - self[f'analysis{i}'](inputs)

        return x, z

    def forward(self, input, cond):
        y2, z2_quant, h2_quant, z_likelihood, h_likelihood, rx \
            = self.encode(input, 0, 0, cond)
        decoded_x, decoded_z = self.decode(cond, z2_quant, cond, rx=rx)
        decoded_x = self.QE(decoded_x)

        return {
            "x_hat": decoded_x,
            "likelihoods": {"z": z_likelihood, "h": h_likelihood},
        }

    def compress(self, input, cond):
        y2, z2_quant, h2_quant, stream_z, stream_h, rx \
            = self.encode(input, 0, 0, cond, compress=True)

        decoded_x, decoded_z = self.decode(cond, z2_quant, cond, rx=rx)
        decoded_x = self.QE(decoded_x)
        return {
            "x_hat": decoded_x,
            "streams": {"z": stream_z, "h": stream_h},
        }

    def decompress(self, strings, cond, shapes):
        stream, side_stream = strings
        h, w = shapes
        h_shape = [1, self.num_hyperpriors, h//64, w//64]
        z_shape = [1, self.num_features, h//16, w//16]

        h_hat = self.entropy_bottleneck.decompress(side_stream, h_shape) 
        hp_feat = self.hyper_synthesis(h_hat)

        pred_feat = self.pred_prior(cond)
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        z_hat = self.conditional_bottleneck.decompress(stream, z_shape, condition=condition)
        decoded_x, decoded_z = self.decode(cond, z_hat, cond)
        decoded_x = self.QE(decoded_x)
        return {
            "x_hat": decoded_x
        }


############################################## for ANFIC ###########################################
class AugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):

    def __init__(
            self, in_channels, num_filters, num_features, num_hyperpriors, kernel_size,
            num_layers=1, hyper_filters=192, num_DQ_layer=6, num_DQ_ch=64, 
            use_mean=False, use_context=False, simplify_gdn=False, 
            condition='Gaussian', quant_mode='noise'
        ):
        super(AugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)

        self.num_layers = num_layers    
        self.num_features = num_features
        self.num_hyperpriors = num_hyperpriors
        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        # analysis and synthesis pairs
        for i in range(num_layers):
            ks = kernel_size[i]
            self.add_module('analysis'+str(i), ANATransform(in_channels, num_features, num_filters, ks, simplify_gdn=simplify_gdn))
            self.add_module('synthesis'+str(i), SYNTransform(in_channels, num_features, num_filters, ks, simplify_gdn=simplify_gdn))

        # hyper prior
        self.hyper_analysis = GoogleHyperAnalysisTransform(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, hyper_filters, num_hyperpriors)

        # quality enhancement network
        self.QE = DeQuantizationModule(in_channels, in_channels, num_DQ_ch, num_DQ_layer)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, x, z, h, compress=False):
        for i in range(self.num_layers):
            z = z + self[f'analysis{i}'](x)

            if i < self.num_layers - 1:
                x = x - self[f'synthesis{i}'](z)

        h = h + self.hyper_analysis(z.abs() if self.use_abs else z)
        if compress:
            stream_h, h_quant = self.entropy_bottleneck.compress(h, return_sym=True)
        else:
            h_quant, h_likelihood = self.entropy_bottleneck(h)
        condition = self.hyper_synthesis(h_quant)
        
        if compress:
            self.conditional_bottleneck.to('cpu')
            stream_z, z_quant = self.conditional_bottleneck.compress(z, condition=condition, return_sym=True)
            self.conditional_bottleneck.to(condition.device)
        else:
            z_quant, z_likelihood = self.conditional_bottleneck(z, condition=condition)
            
        rx = self[f'synthesis{self.num_layers-1}'](z_quant)
        x = x - rx
        
        if compress:
            return x, z_quant, h_quant, stream_z, stream_h, rx
        else:
            return x, z_quant, h_quant, z_likelihood, h_likelihood, rx

    def decode(self, x, z, rx=None):
        for i in range(self.num_layers-1, -1, -1):           
            if (i == self.num_layers - 1) and isinstance(rx, torch.Tensor):
                x = x + rx
            else:
                x = x + self[f'synthesis{i}'](z)

            if i:
                z = z - self[f'analysis{i}'](x)

        return x, z

    def forward(self, input):
        y2, z2_quant, h2_quant, z_likelihood, h_likelihood, rx \
            = self.encode(input, 0, 0)
        decoded_x, decoded_z = self.decode(0, z2_quant, rx=rx)
        decoded_x = self.QE(decoded_x)

        return {
            "x_hat": decoded_x,
            "likelihoods": {"z": z_likelihood, "h": h_likelihood},
        }
        
    def compress(self, input):
        y2, z2_quant, h2_quant, stream_z, stream_h, rx \
            = self.encode(input, 0, 0, compress=True)

        decoded_x, decoded_z = self.decode(0, z2_quant, rx=rx)
        decoded_x = self.QE(decoded_x)
        return {
            "x_hat": decoded_x,
            "streams": {"z": stream_z, "h": stream_h},
        }

    def decompress(self, strings, shapes):
        stream, side_stream = strings
        h, w = shapes
        h_shape = [1, self.num_hyperpriors, h//64, w//64]
        z_shape = [1, self.num_features, h//16, w//16]
        
        h_hat = self.entropy_bottleneck.decompress(side_stream, h_shape) 
        condition = self.hyper_synthesis(h_hat)

        self.conditional_bottleneck.to('cpu')
        z_hat = self.conditional_bottleneck.decompress(stream, z_shape, condition=condition)
        self.conditional_bottleneck.to(condition.device)
        decoded_x, decoded_z = self.decode(0, z_hat)
        decoded_x = self.QE(decoded_x)
        return {
            "x_hat": decoded_x
        }


__CODER_TYPES__ = {
    "ANFHyperPriorCoder": AugmentedNormalizedFlowHyperPriorCoder,
    "BCANFCodec": BCANFCodec
}


def get_coder_from_args(args):
    coder = __CODER_TYPES__[args.architecture]   

    kwargs = vars(args)
    required_args = signature(coder).parameters.keys()
    
    used_kwargs = {}
    for k, v in kwargs.items():
        if k in required_args:
            used_kwargs[k] = v

    return partial(coder, **used_kwargs)
