import torch
from functools import partial
from typing import Union, Callable
import ot as pot
import warnings

class OTSampler(object):
    '''
    Minibatch OT as proposed by Tong et al. in https://arxiv.org/pdf/2302.00482

    This class will compute and return optimal transport pairs from two x_0 and x_1
    '''
    def __init__(
        self, method:str, metric:Callable[[torch.tensor, torch.tensor], torch.tensor],
        num_threads:Union[int, str]=1, reg:float=0.05, reg_m:float=1.0, normalize_cost=True
    ):
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.normalize_cost = True

    def _get_map(self, x_0, x_1):
        '''
        Compute OT sample plan based on distance metric
        :param x_0:
            x at time 0 to connect to x_1.
        :type x_0: torch.tensor
        :param x_1:
            x at time 1 to connect to x_0.
        :type x_1: torch.tensor
        '''
        a, b = pot.unif(x_0.shape[0]), pot.unif(x_1.shape[0])
        assert x_0.shape == x_1.shape
        distance = self.metric(x_0, x_1)
        if self.normalize_cost:
            distance /= distance.max()
        p = self.ot_fn(a, b, distance.detach.cpu())
        if not torch.all(torch.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", distance.mean(), distance.max())
            print(x_0, x_1) 

        if torch.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = torch.ones_like(p) / p.size
        return p