import torch

import numpy as np
from normflows.flows.affine.coupling import Flow
from scipy import optimize

import commons


MIN_NR_MCMC_SAMPLES = 0

class TrainableLOFTLayer(Flow):

    def __init__(self, dim, initial_t, train_t):
        assert(initial_t >= 1.0)
        super().__init__()
        self.dim = dim
        self.rep_t = torch.ones(dim) * (initial_t - 1.0) # reparameterization of t 
        self.rep_t = commons.moveToDevice(self.rep_t)
        self.rep_t = torch.nn.Parameter(self.rep_t, requires_grad=train_t)
        return

    def forward(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # batch size
        assert(z.shape[1] >= 10) # dimension

        t = self.get_t()

        new_value, part1 = TrainableLOFTLayer.LOFT_forward_static(t, z)

        log_derivatives = - torch.log(part1 + 1.0)

        log_det = torch.sum(log_derivatives, axis = 1)

        return new_value, log_det
    
    def get_t(self):
        return 1.0 + torch.nn.functional.softplus(self.rep_t)

    def LOFT_forward_static(t, z):
        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)

        new_value = torch.sign(z) * (torch.log(part1 + 1) + part2)

        return new_value, part1
    
    def inverse(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # Monte Carlo Samples
        assert(z.shape[1] >= 10) # dimension

        t = self.get_t()

        part1 = torch.max(torch.abs(z) - t, torch.tensor(0.0))
        part2 = torch.min(torch.abs(z), t)
        
        new_value = torch.sign(z) * (torch.exp(part1) - 1.0 + part2)
        
        log_det = torch.sum(part1, axis = 1)

        return new_value, log_det


class ScaledTanhLayer(Flow):
    
    def __init__(self, dim, t):
        super().__init__()
        self.dim = dim
        self.s = torch.tensor(ScaledTanhLayer.get_optimal_scaling(torch.tensor(t)))
        self.s = commons.moveToDevice(self.s)
        print("ScaledTanhLayer s = ", self.s)
        return
    

    # set scaling parameter s such that 
    # derivative at 0 is 1 and 
    # LOFT(2.0 * t) = scaled_tanh(2.0 * t)
    def get_optimal_scaling(t):
        t1 = 2.0 * t.item()
        t2 = TrainableLOFTLayer.LOFT_forward_static(t, 2.0 * t)[0].item()

        def f(s):
            return s * np.tanh((1.0 / s) * t1) - t2

        lb = 1.0
        rb = 2.0 * t.item()

        # print("t1 = ", t1)
        # print("t2 = ", t2)
        # print("left_bound_value = ", f(lb))
        # print("right_bound_value = ", f(rb))

        solution = optimize.root_scalar(f, bracket = [lb, rb], method='brentq')
        return solution.root
    
    
    def forward(self, z):
        assert(self.s.requires_grad == False)
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # batch size
        assert(z.shape[1] >= 10) # dimension

        tanh_part = torch.tanh(z / self.s)
        new_value = self.s * tanh_part

        log_derivatives = torch.log(1.0 - torch.square(tanh_part))

        log_det = torch.sum(log_derivatives, axis = 1)
        
        return new_value, log_det

    def inverse(self, z):
        assert(self.s.requires_grad == False)
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # Monte Carlo Samples
        assert(z.shape[1] >= 10) # dimension
        
        new_value = self.s * torch.atanh(z / self.s)

        log_derivatives = - torch.log(1.0 - torch.square(z / self.s))

        log_det = torch.sum(log_derivatives, axis = 1)
        
        return new_value, log_det
    

class TruncationLayer(Flow):

    def __init__(self, dim, threshold):
        assert(threshold >= 1.0)
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        return

    def forward(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # batch size
        assert(z.shape[1] >= 10) # dimension

        z[torch.abs(z) > self.threshold] = torch.nan

        # assert(not z.isnan().any())

        new_value = z
        log_det = 0.0
        return new_value, log_det
    
    
    def inverse(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # Monte Carlo Samples
        assert(z.shape[1] >= 10) # dimension
        
        new_value = z
        log_det = 0.0

        return new_value, log_det


class PositiveConstraintLayer(Flow):

    def __init__(self, pos_contraint_ids, total_dim):
        super().__init__()
        assert(torch.is_tensor(pos_contraint_ids))

        self.total_dim = total_dim
        self.pos_contraint_ids = pos_contraint_ids
        self.no_contraint_ids = torch.tensor(np.delete(np.arange(total_dim), pos_contraint_ids))

        test = np.zeros(total_dim)
        test[self.no_contraint_ids] += 1
        test[self.pos_contraint_ids] += 1
        assert(np.all(test == 1))
        assert(len(self.pos_contraint_ids.shape) == len(self.no_contraint_ids.shape))
        return

    # checked
    def forward(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # Monte Carlo Samples
        assert(z.shape[1] == self.total_dim) # dimension
        new_values_pos, log_det_each_dim = self.forward_one_dim(z[:, self.pos_contraint_ids])
        
        assert(new_values_pos.shape[0] >= MIN_NR_MCMC_SAMPLES)
        assert(new_values_pos.shape[1] == self.pos_contraint_ids.shape[0])
        
        all_new_values = torch.zeros_like(z)
        all_new_values[:, self.no_contraint_ids] = z[:, self.no_contraint_ids]
        all_new_values[:, self.pos_contraint_ids] = new_values_pos
        
        assert(log_det_each_dim.shape[0] >= MIN_NR_MCMC_SAMPLES)
        assert(log_det_each_dim.shape[1] == self.pos_contraint_ids.shape[0])
        log_det = torch.sum(log_det_each_dim, axis = 1)
        assert(log_det.shape[0] >= MIN_NR_MCMC_SAMPLES)

        return all_new_values, log_det


    def inverse(self, z):
        assert(z.shape[0] >= MIN_NR_MCMC_SAMPLES) # Monte Carlo Samples
        assert(z.shape[1] == self.total_dim) # dimension
        assert(torch.all(z[:, self.pos_contraint_ids] >= 0.0))
        new_value_posOrNeg, log_det_each_dim = self.inverse_one_dim(z[:, self.pos_contraint_ids])
        
        assert(new_value_posOrNeg.shape[0] >= MIN_NR_MCMC_SAMPLES)
        assert(new_value_posOrNeg.shape[1] == self.pos_contraint_ids.shape[0])
        all_new_values = torch.zeros_like(z)
        all_new_values[:, self.no_contraint_ids] = z[:, self.no_contraint_ids]
        all_new_values[:, self.pos_contraint_ids] = new_value_posOrNeg
        
        assert(log_det_each_dim.shape[0] >= MIN_NR_MCMC_SAMPLES)
        assert(log_det_each_dim.shape[1] == self.pos_contraint_ids.shape[0])
        log_det = torch.sum(log_det_each_dim, axis = 1)
        assert(log_det.shape[0] >= MIN_NR_MCMC_SAMPLES)
        
        return all_new_values, log_det


    def forward_one_dim(self, z):

        new_value = torch.nn.functional.softplus(z)
        log_derivative = z - new_value

        return new_value, log_derivative


    def inverse_one_dim(self, z):

        new_value = torch.log(torch.special.expm1(z))
        log_derivative = z - new_value
        
        return new_value, log_derivative

def leakyClamp(value, minAlpha, maxAlpha):
    reLU = torch.nn.LeakyReLU()
    leftTruncation = reLU(value + minAlpha) - minAlpha
    rightTruncation = - (reLU(-leftTruncation + maxAlpha) - maxAlpha)
    return rightTruncation


def softClampAsymNew(value, alpha, minValue):
    reLU = torch.nn.ReLU()
    posValues = (2.0 * alpha / torch.pi) * torch.arctan(reLU(value) / alpha)
    negValues = - (2.0 / torch.pi) * reLU(-value)
    values = negValues + posValues
    return torch.clamp(values,  min = minValue)


def softClampAsymAdvanced(value, negAlpha, posAlpha):
    reLU = torch.nn.ReLU()
    posValues = (2.0 * posAlpha / torch.pi) * torch.arctan(reLU(value) / posAlpha)
    negValues = (2.0 * negAlpha / torch.pi) * torch.arctan(-reLU(-value) / negAlpha)
    return negValues + posValues


def softClamp(value, alpha):
    return (2.0 * alpha / torch.pi) * torch.arctan(value / alpha)


def softClampAsymAdvanced_differentImpl(value, negAlpha, posAlpha):
    posValues = 0.5 * (torch.sign(value) + 1.0)
    negValues = 0.5 * (-torch.sign(value) + 1.0)

    posValues = posValues * (2.0 * posAlpha / torch.pi) * torch.arctan(value / posAlpha)
    negValues = negValues * (2.0 * negAlpha / torch.pi) * torch.arctan(value / negAlpha)
    return negValues + posValues


class MaskedAffineFlowThresholded(Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None, threshold=None, variation=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        assert(variation is not None)
        assert(threshold is not None)
        assert(threshold >= 0.05)

        self.variation = variation
        self.threshold = threshold

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

        
        self.exp_threshold = commons.moveToDevice(torch.exp(torch.tensor(self.threshold)))
        return

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale, trans = self.limit(scale, trans)

        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det

    def limit(self, scale, trans):
        
        if self.variation == "var17":
            # RealNVP variation as proposed in "GUIDED IMAGE GENERATION WITH CONDITIONAL INVERTIBLE NEURAL NETWORKS" https://arxiv.org/pdf/1907.02392.pdf
            scale = softClampAsymAdvanced_differentImpl(scale, negAlpha = self.threshold, posAlpha = self.threshold)
        elif self.variation == "var19":
            # proposed clippling method
            scale = softClampAsymAdvanced_differentImpl(scale, negAlpha = 2.0, posAlpha = self.threshold)
        elif self.variation == "var20":
            # proposed clippling method
            scale = softClampAsymAdvanced_differentImpl(scale, negAlpha = 5.0, posAlpha = self.threshold)
        elif self.variation == "tanh":
            # used by ATAF method
            scale = torch.tanh(scale)
        else:
            assert(False)

        return scale, trans
    

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale, trans = self.limit(scale, trans)

        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det




class MaskedAffineFlowSoftClamp(Flow):
    """RealNVP variation as proposed in "GUIDED IMAGE GENERATION WITH CONDITIONAL INVERTIBLE NEURAL NETWORKS"
    https://arxiv.org/pdf/1907.02392.pdf
    """

    def __init__(self, b, t=None, s=None, alpha=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        self.alpha = alpha  # previous works recommend 1.9

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)


    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale = softClamp(scale, self.alpha)

        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det


    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        scale = softClamp(scale, self.alpha)

        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det
    


def squashedSigmoid(value):
    return 0.5 * torch.sigmoid(value) + 0.5

class MaskedAffineFlowSquashedSigmoid(Flow):
    """RealNVP variation as proposed in "Understanding and Mitigating Exploding Inverses in Invertible Neural Networks"
    https://proceedings.mlr.press/v130/behrmann21a.html
    """

    def __init__(self, b, t=None, s=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)


    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        z_ = z_masked + (1 - self.b) * (z * squashedSigmoid(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))

        return z_, log_det


    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)

        z_ = z_masked + torch.div((1 - self.b) * (z - trans), squashedSigmoid(scale))
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

