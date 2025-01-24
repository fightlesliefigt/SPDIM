import torch
import torch.nn as nn
from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
from torch.functional import Tensor
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from .manifolds import SymmetricPositiveDefinite
from . import functionals


class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        pass

# %% base classes

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta = 1.0, eta_test = 0.1, test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.ModuleDict()

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer : BaseBatchNorm, domain : Tensor):
        self.batchnorm[str(domain.item())] = layer

    def get_domain_obj(self, domain : Tensor):
        return self.batchnorm[domain.item()]

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        self.batchnorm[str(domain.item())].initrunningstats(X)

    def forward_domain_(self, X, domain,parameter_t,fm_mean):
        res = self.batchnorm[str(domain.item())](X,parameter_t,fm_mean)
        return res

    def forward(self, X, d,parameter_t,fm_mean):
        du = d.unique()

        X_normalized = torch.empty_like(X)
        res = [(self.forward_domain_(X[d==domain], domain,parameter_t,fm_mean),torch.nonzero(d==domain))
                for domain in du]
        X_out, ixs = zip(*res)
        X_out, ixs = torch.cat(X_out), torch.cat(ixs).flatten()
        X_normalized[ixs] = X_out
        
        return X_normalized


# %% SPD manifold implementation

class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim : int, 
                 eta = 1., eta_test = 0.1,
                 karcher_steps : int = 1, learn_mean = True, learn_std = True, 
                 dispersion : BatchNormDispersion = BatchNormDispersion.SCALAR, 
                 eps = 1e-5, mean = None, std = None, **kwargs):
        super().__init__(eta, eta_test)
        # the last two dimensions are used for SPD manifold
        assert(shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps
        self.manifold = SymmetricPositiveDefinite()
        
        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        init_var = torch.ones((*shape[:-2], 1), **kwargs)

        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', init_mean)
        self.register_buffer('running_var_test', init_var)

        if mean is not None:
            self.mean = mean
        else:
            if self.learn_mean:
                self.mean = ManifoldParameter(init_mean.clone(), manifold=SymmetricPositiveDefinite())
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=SymmetricPositiveDefinite())
        
        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()

    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean, geom_dist = functionals.spd_mean_kracher_flow(X, dim=self.batchdim, return_dist=True)
        self.running_mean_test = self.running_mean.clone()
        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = geom_dist.square().mean(dim=self.batchdim, keepdim=True).clamp(min=functionals.EPS[X.dtype])[...,None]
            self.running_var_test = self.running_var.clone()


    def forward(self, X,parameter_t,fm_mean):
        if self.training:
            # compute the Karcher flow for the current batch
            batch_mean = X.mean(dim=self.batchdim, keepdim=True)
            if X.shape[self.batchdim] > 1:
                for _ in range(self.karcher_steps):
                    bm_sq, bm_invsq = functionals.sym_invsqrtm2.apply(batch_mean.detach())
                    XT = functionals.sym_logm.apply(bm_invsq @ X @ bm_invsq)
                    GT = XT.mean(dim=self.batchdim, keepdim=True)
                    batch_mean = bm_sq @ functionals.sym_expm.apply(GT) @ bm_sq
            
            # update the running mean
            rm = functionals.spd_2point_interpolation(self.running_mean, batch_mean, self.eta)

            if self.dispersion is BatchNormDispersion.SCALAR:
                if X.shape[self.batchdim] > 1:
                    GT = functionals.sym_logm.apply(bm_invsq @ rm @ bm_invsq)
                    batch_var = torch.norm(XT - GT, p='fro', dim=(-2,-1), keepdim=True).square().mean(dim=self.batchdim, keepdim=True).squeeze(-1)
                else:
                    rm_invsq = functionals.sym_invsqrtm.apply(rm)
                    rminvX = rm_invsq @ batch_mean @ rm_invsq
                    batch_var = functionals.sym_logm.apply(rminvX).square().sum(dim=(-1,-2), keepdim=True).squeeze(-1)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var
            
            # rescale to desired dispersion
            if self.dispersion is BatchNormDispersion.SCALAR:
                Xn = self.manifold.rescale_transp_geosedic_identity_transp(X,rm,self.std/(rv + self.eps).sqrt(),parameter_t)
            else:
                Xn = self.manifold.transp_geosedic_identity_transp(X,rm,parameter_t)
        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            # if there is input for the mean, use it
            if fm_mean is None:
                rm = self.running_mean_test
            else:
                rm = fm_mean
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test
                
            Xn = self.manifold.transp_geosedic_identity_transp(X,rm,parameter_t)

        if self.training:
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = functionals.spd_2point_interpolation(self.running_mean_test, batch_mean, self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    if X.shape[self.batchdim] > 1:
                        GT_test = functionals.sym_logm.apply(bm_invsq @ self.running_mean_test @ bm_invsq)
                        batch_var_test = torch.norm(XT - GT_test, p='fro', dim=(-2,-1), keepdim=True).square().mean(dim=self.batchdim, keepdim=True).squeeze(-1)
                    else:
                        rm_invsq = functionals.sym_invsqrtm.apply(self.running_mean_test)
                        rminvX = rm_invsq @ batch_mean @ rm_invsq
                        batch_var_test = functionals.sym_logm.apply(rminvX).square().sum(dim=(-1,-2), keepdim=True).squeeze(-1)
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test

        return Xn

class AdaMomSPDBatchNorm(SPDBatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive momentum batch normalization on the SPD manifold.

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)


class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    """
    Domain-specific batch normalization on the SPD manifold 

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    """

    domain_bn_cls = None # needs to be overwritten by subclasses

    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim :int,
                 learn_mean : bool = True, learn_std : bool = True,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 eta = 1., eta_test = 0.1, domains : Tensor = Tensor([]), **kwargs):
        super().__init__()

        assert(shape[-1] == shape[-2])

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        init_mean = torch.diag_embed(torch.ones(shape[:-1], **kwargs))
        if self.learn_mean:
            self.mean = ManifoldParameter(init_mean, 
                                        manifold=SymmetricPositiveDefinite())
        else:
            self.mean = ManifoldTensor(init_mean, 
                                       manifold=SymmetricPositiveDefinite())
        
        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((*shape[:-2], 1), **kwargs)
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
            else:
                self.std = init_var.clone()
        else:
            self.std = None

        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(cls(shape=shape, batchdim=batchdim, 
                                learn_mean=learn_mean,learn_std=learn_std, dispersion=dispersion,
                                mean=self.mean, std=self.std, eta=eta, eta_test=eta_test, **kwargs),
                            domain)

        self.set_test_stats_mode(test_stats_mode)


class AdaMomDomainSPDBatchNorm(DomainSPDBatchNormImpl):
    """
    Combines domain-specific batch normalization on the SPD manifold
    with adaptive momentum batch normalization.

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    The momentum terms can be controlled with a momentum scheduler.
    """

    domain_bn_cls = AdaMomSPDBatchNorm
