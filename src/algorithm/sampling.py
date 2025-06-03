import numpy as np
import torch
from copy import deepcopy
from abc import abstractmethod
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.calculate_crowding_distance import calc_crowding_distance



import ray 
@ray.remote(num_cpus=1)
def evaluate_placer(placer, x0):
    return placer.evaluate(x0)



class BasicSampling():
    def __init__(self, args, evaluator, record_func) -> None:
        self.args = args
        self.evaluator = evaluator
        self.n_repeat = self.args.n_sampling_repeat
        self.eval_metrics = args.eval_metrics
    
        self.record_func = record_func
    
    def do(self, n_samples):
        X, Y = None, None
        macro_pos = None

        X, y, macro_pos = self._sampling_do(
            n_samples=n_samples * self.n_repeat
        )
        Y = np.column_stack(
                [y[metric] for metric in self.eval_metrics],
            )
        
        nds = NonDominatedSorting()
        fronts = nds.do(Y)

        selected_indices = []
        front_idx = 0

        while len(selected_indices) < n_samples and front_idx < len(fronts):
            current_front = fronts[front_idx]

            if len(selected_indices) + len(current_front) <= n_samples:
                selected_indices.extend(current_front)
            else:
                remaining = n_samples - len(selected_indices)

                crowding_dist = calc_crowding_distance(Y[current_front])

                selected_from_front = current_front[
                    np.argsort(-crowding_dist)[:remaining]
                ]
                selected_indices.extend(selected_from_front)

            front_idx += 1

        selected_indices = np.array(selected_indices)
        remaining = ~np.isin(np.arange(X.shape[0]), selected_indices)
        assert np.sum(remaining) == (self.n_repeat - 1) * n_samples
        
        if self.n_repeat > 1:
            self.record_func(
                Y=Y[remaining], 
                macro_pos_all=list(np.array(macro_pos)[remaining])
            ) 

        X = X[selected_indices]
        return X
    
    @abstractmethod
    def _sampling_do(self, n_samples, **kwargs):
        pass


###################################################################
#  Grid Guide sampling
###################################################################

class GGRandomSampling(BasicSampling):
    def __init__(self, args, evaluator, record_func) -> None:
        BasicSampling.__init__(self, args=args, evaluator=evaluator, record_func=record_func)

        self.xl = evaluator.xl
        self.xu = evaluator.xu
        self.n_dim = evaluator.n_dim
    
    def _sampling_do(self, n_samples):
        x = np.column_stack(
            [np.random.randint(self.xl[k], self.xu[k] + 1, size=n_samples) 
            for k in range(self.n_dim)]
        )
        y, macro_pos = self.evaluator.evaluate(x)

        return x, y, macro_pos
    
###################################################################
#  SP sampling
###################################################################

class SPRandomSampling(BasicSampling):
    def __init__(self, args, evaluator, record_func) -> None:
        BasicSampling.__init__(self, args=args, evaluator=evaluator, record_func=record_func)

        self.n_dim = evaluator.n_dim
    
    def _sampling_do(self, n_samples):
        x = self._permutation_sample(self, n_samples)
        y, macro_pos = self.evaluator.evaluate(x)
         
        return x, y, macro_pos

    def _permutation_sample(self, n_samples):
        X1 = self._get_permutation(n_samples=n_samples)
        X2 = self._get_permutation(n_samples=n_samples)
        X = np.concatenate([X1, X2], axis=1)
        return X
    
    def _get_permutation(self, n_samples):
        n_dim = self.n_dim // 2
        X = np.full((n_samples, n_dim), 0, dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(n_dim)
        return X
    
REGISTRY = {}
REGISTRY["gg"] = GGRandomSampling
REGISTRY["sp"] = SPRandomSampling