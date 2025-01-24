import torch
import spdnets.modules as modules
import spdnets.batchnorm as bn
from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam
from typing import Optional, Union



class TSMNet(torch.nn.Module):
    def __init__(self, temporal_filters, 
                 spatial_filters = 40,
                 subspacedims = 20,
                 temp_cnn_kernel = 25,
                 bnorm_dispersion : Union[str, bn.BatchNormDispersion] = bn.BatchNormDispersion.SCALAR,
                 nclasses : Optional[int] = None,
                 nchannels : Optional[int] = None,
                 nsamples : Optional[int] = None,
                 domain_adaptation : bool = True,
                 domains = [],
                 spd_device = 'cpu',
                 spd_dtype = torch.double,
        ):
        super().__init__()

        self.nclasses_ = nclasses
        self.nchannels_ = nchannels
        self.nsamples_ = nsamples
        self.temporal_filters_ = temporal_filters
        self.spatial_filters_ = spatial_filters
        self.subspacedimes = subspacedims
        self.spd_device_ = torch.device(spd_device)
        self.spd_dtype_ = spd_dtype
        self.domains_ = domains
        self.domain_adaptation_ = domain_adaptation

        if isinstance(bnorm_dispersion, str):
            self.bnorm_dispersion_ = bn.BatchNormDispersion[bnorm_dispersion]
        else:
            self.bnorm_dispersion_ = bnorm_dispersion
        
        tsdim = int(subspacedims*(subspacedims+1)/2)
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.temporal_filters_, kernel_size=(1,temp_cnn_kernel),
                            padding='same', padding_mode='reflect'),
            torch.nn.Conv2d(self.temporal_filters_, self.spatial_filters_,(self.nchannels_, 1)),
            torch.nn.Flatten(start_dim=2),
        )

        self.cov_pooling = torch.nn.Sequential(
            modules.CovariancePool(),
        )
        
        if self.domain_adaptation_:
            self.spdbnorm = bn.AdaMomDomainSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                domains=self.domains_,
                                learn_mean=False,learn_std=True, 
                                dispersion=self.bnorm_dispersion_, 
                                eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)
        else:
            self.spdbnorm = bn.AdaMomSPDBatchNorm((1,subspacedims,subspacedims), batchdim=0, 
                                          dispersion=self.bnorm_dispersion_, 
                                          learn_mean=False,learn_std=True,
                                          eta=1., eta_test=.1, dtype=self.spd_dtype_, device=self.spd_device_)

        self.spdnet = torch.nn.Sequential(
            modules.BiMap((1,self.spatial_filters_,subspacedims), dtype=self.spd_dtype_, device=self.spd_device_),
            modules.ReEig(threshold=1e-4),
        )
        self.logeig = torch.nn.Sequential(
            modules.LogEig(subspacedims),
            torch.nn.Flatten(start_dim=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(tsdim,self.nclasses_),
        )

    def to(self, device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[int, torch.dtype]] = None, non_blocking: bool = False):
        self.cnn = self.cnn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.classifier = self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, inputs, domains, parameter_t,fm_mean):
        h = self.cnn(inputs[:,None,...])
        C = self.cov_pooling(h).to(self.spdnet[0].W)
        l = self.spdnet(C)
        self.spdnet_data=l
        l = self.spdbnorm(l,domains, parameter_t,fm_mean) if self.domain_adaptation_ else self.spdbnorm(l)
        l = self.logeig(l)
        l = l.to(inputs)
        outputs = self.classifier(l)
        return outputs

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-4):

        params = []
        zero_wd_params = []
        
        for name, param in self.named_parameters():
            if name.startswith('spdnet') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            elif name.startswith('spdbn') and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        
        pgroups = [
            dict(params = zero_wd_params, weight_decay=0.),
            dict(params = params)
        ]

        return RiemannianAdam(pgroups, lr=lr, weight_decay=weight_decay)


    def domainadapt_finetune(self, x, y, d,mode):
        if self.domain_adaptation_:
            if mode == 'refit':
                self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
                with torch.no_grad():
                    for du in d.unique():
                        self.forward(x[d==du], d[d==du],parameter_t=1,fm_mean=None)
                self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)                        
            else:
                raise ValueError(f"Unknown mode {mode}")

    def get_spdnet_data(self):
        return self.spdnet_data
    


    
