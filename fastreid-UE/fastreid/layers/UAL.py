import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F

import math

"""
Adapted from UAL by Andreas Gebhardt in 2024.
"""

# TODO: change the init to fastreid's init?

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl


class Bayes_Gaussion_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None, **kwargs):

        super(Bayes_Gaussion_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = 1.0

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.01,
                'posterior_mu_initial': (0, 0.01),
                'posterior_rho_initial': (-4, 0.01),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_test = Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
            self.bias_test = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_rho', None)

        self.train_sample_nums = 5
        self.test_sample_nums = 5

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def reset_weight(self):
        weight_test, bias_test = self.sample_weight(self.test_sample_nums)
        self.weight_test.zero_()
        self.weight_test += weight_test
        if self.bias is not None:
            self.bias_test.zero_()
            self.bias_test += bias_test
        else:
            self.bias_test = None

    def sample_weight(self, nums):
        weight, bias = 0, None

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        normal = torch.distributions.Normal(loc=self.weight, scale=self.scale * weight_sigma)
        for i in range(nums):
            weight += normal.rsample()
        weight /= nums

        if self.bias is not None:
            bias = 0
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            normal = torch.distributions.Normal(loc=self.bias, scale=self.scale * bias_sigma)
            for i in range(nums):
                bias += normal.rsample()
            bias /= nums
        return weight, bias

    def forward(self, input):
        if self.training or True: # do without the extra stuff because it makes no difference
            weight, bias = self.sample_weight(self.train_sample_nums)
        else:
            # weight, bias = self.sample_weight(self.test_sample_nums)
            weight, bias = self.weight_test, self.bias_test
            # weight, bias = self.weight, self.bias
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.weight, weight_sigma)
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias, bias_sigma)
        return kl


class Bayes_Dropout_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, prob=0.5):

        super(Bayes_Dropout_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prob = prob

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_test = Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_test = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.train_sample_nums = 1
        self.test_sample_nums = 1

        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_weight(self):
        weight_test, bias_test = self.sample_weight(self.test_sample_nums)
        self.weight_test.zero_()
        self.weight_test += weight_test
        if self.bias is not None:
            self.bias_test.zero_()
            self.bias_test += bias_test
        else:
            self.bias_test = None

    def sample_weight(self, nums):
        weight = 0
        if self.bias is not None:
            bias = 0
        for i in range(nums):
            bernolli = torch.distributions.Bernoulli(probs=self.prob)
            weight += self.weight * bernolli.sample(self.weight.shape).to(self.weight.device)
            weight *= 0.7 / self.prob
            if self.bias is not None:
                bias += self.bias * bernolli.sample(self.bias.shape).to(self.bias.device)
                bias *= 0.7 / self.prob
            else:
                bias = None
        weight /= nums
        if self.bias is not None:
            bias /= nums
        return weight, bias

    def forward(self, input):
        if self.training or True: # do without the extra stuff because it makes no difference
            weight, bias = self.sample_weight(self.train_sample_nums)
        else:
            weight, bias = self.weight_test, self.bias_test
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        return torch.zeros(1).cuda()


class BayesDropoutConv2_5d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=0, dilation=1, prob=0.7):
        super().__init__()

        self.distribution = torch.distributions.Bernoulli(probs=prob) # no need to re-instantiate this every forward pass
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.use_bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_channels
        self.prob = prob

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset_parameters()


    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input): # input shape (B,C,D,H,W)

        #print(input.shape)

        B,C,D,H,W = input.shape 

        # stack along channel dimension instead of new dimension
        input = input.permute(0,2,1,3,4).reshape(B, D*C, H, W) # yes, we need the permute.

        #torch.cat(torch.unbind(input, dim=2),dim=1) # tested identical to permute/reshape solution above # TODO: benchmark which is faster

        stacked_weight = self.weight.repeat(D, 1, 1, 1) # each group uses this weight # (ch_out * D, ch_in, k_H, k_W)
        mask = self.distribution.sample(stacked_weight.shape).to(self.weight.device)
        #dist = torch.distributions.Bernoulli(probs=self.prob)
        #mask = dist.sample(stacked_weight.shape).to(self.weight.device)
        final_weight = stacked_weight * mask
        #final_weight = stacked_weight 

        final_bias = None
        if self.use_bias:
            stacked_bias = self.bias.repeat(D)
            bias_mask = self.distribution.sample(stacked_bias.shape).to(self.bias.device)
            #bias_mask = dist.sample(stacked_bias.shape).to(self.bias.device)
            final_bias = stacked_bias * bias_mask

        # grouped convolution: 
        # - we split the input channels into D groups of size C
        # - each such group is convolved with a filter of according size (normal conv2D on one of the D-slices of the input)
        # - the result is one of the output channels
        # - the first of D equally sized output channel groups work with the first C input channels, the second the second, etc.
        output = F.conv2d(input, final_weight, final_bias if self.use_bias else None,
                              self.stride, self.padding, self.dilation,   groups=D  ) # TODO: it would be preferrable if there was a dedicated convolution with two channel dimensions over one of which we could use groups
        # output shape: (B, ch_out*D, H', W')

        return output.reshape(B, D, self.out_channels, output.shape[-2], output.shape[-1]).permute(0,2,1,3,4) # return to input format


class BayesGaussianConv2_5d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=0, dilation=1, prob=0.7, priors=None):
        super().__init__()

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.use_bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_channels

        self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_stddev_raw = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_stddev_raw = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None) 
            self.register_parameter('bias_stddev_raw', None) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.01,
                'posterior_mu_initial': (0, 0.01),
                'posterior_rho_initial': (-4, 0.01),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.reset_parameters()


    def reset_parameters(self):
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_stddev_raw.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_stddev_raw.data.normal_(*self.posterior_rho_initial)


    def forward(self, input): 

        B,C,D,H,W = input.shape

        input = input.permute(0,2,1,3,4).reshape(B, D*C, H, W) # stack along channel dimension instead of new dimension

        stacked_weight_mu = self.weight_mu.repeat(D, 1, 1, 1)
        stacked_weight_stddev = torch.log1p(torch.exp(self.weight_stddev_raw)).repeat(D, 1, 1, 1)
        eps = torch.empty(stacked_weight_mu.shape, dtype=self.weight_mu.dtype, device=self.weight_mu.device).normal_()
        final_weight = stacked_weight_mu + eps * stacked_weight_stddev # manual reparametrization trick

        final_bias = None
        if self.use_bias:
            stacked_bias_mu = self.bias_mu.repeat(D)
            stacked_bias_stddev = torch.log1p(torch.exp(self.bias_stddev_raw)).repeat(D)
            bias_eps = torch.empty(stacked_bias_mu.shape, dtype=self.bias_mu.dtype, device=self.bias_mu.device).normal_()
            final_bias = stacked_bias_mu + bias_eps * stacked_bias_stddev

        # grouped convolution: 
        # - we split the input channels into D groups of size C
        # - each such group is convolved with a filter of according size (normal conv2D on one of the D-slices of the input)
        # - the result is one of the output channels
        # - the first of D equally sized output channel groups work with the first C input channels, the second the second, etc.
        output = F.conv2d(input, final_weight, final_bias if self.use_bias else None,
                              self.stride, self.padding, self.dilation,   groups=D  ) # TODO: it would be preferrable if there was a dedicated convolution with two channel dimensions over one of which we could use groups
        # output shape: (B, ch_out*D, H', W')

        return output.reshape(B, D, self.out_channels, output.shape[-2], output.shape[-1]).permute(0,2,1,3,4) # return to input format