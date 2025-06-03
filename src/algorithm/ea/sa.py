import numpy as np 
import pickle
import time
import os
from pymoo.core.population import Population
from .problem import PlacementProblem
from utils.debug import * 
from utils.constant import INF
from ..basic_algo import BasicAlgo

from src.algorithm.sampling import REGISTRY as SAMPLE_REGISTRY
from .ops import REGISTRY as OPS_REGISTRY


class SA(BasicAlgo):
    def __init__(self, args, evaluator, logger):
        super(SA, self).__init__(args=args, evaluator=evaluator, logger=logger)
        assert len(self.eval_metrics) == 1, self.eval_metrics
        self.node_cnt = evaluator.node_cnt
        
        self.decay = args.decay
        self.init_T = args.T 
        self.T = args.T 
        self.update_freq = args.update_freq
        self.max_evals = args.max_evals
        
        self.population = None
        self.population_metric = INF

        self.problem = PlacementProblem(
            args=args,
            evaluator=evaluator
        )
        
        self.sampling = SAMPLE_REGISTRY[self.args.placer](self.args, evaluator, self._record_results)
        self.mutation = OPS_REGISTRY["mutation"][self.args.placer](self.args)
    
    
    def run(self):
        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            self.T = checkpoint["temperature"]

        while self.n_eval < self.max_evals:
            t_start = time.time()
            if self.population is None:
                if checkpoint is None:
                    now_x = self.sampling.do(1)
                    now_x = Population.new(X=now_x)
                else:
                    now_x = checkpoint["population"]
            else:
                now_x = self.mutation.do(self.problem, self.population, inplace=True)
            
            if self.start_from_checkpoint:
                now_metric = checkpoint["fitness"]

                self.population = now_x
                self.population_metric = now_metric

                self.start_from_checkpoint = False
            else:

                res = {}
                self.problem._evaluate(now_x.get("X"), res)
                now_metric = res["F"].item()
                now_macro_pos = res["macro_pos"]

                if self.population_metric < now_metric:
                    # sa
                    exp_argument = (self.population_metric - now_metric) / self.T 
                    probability = np.exp(exp_argument)
                    if np.random.uniform(0, 1) < probability:
                        self.population = now_x
                        self.population_metric = now_metric
                else:
                    self.population = now_x
                    self.population_metric = now_metric

            
                t_temp = time.time()
                t_eval = t_temp - t_start
                self.t_total += t_eval
                t_each_eval = t_temp - t_start
                avg_t_each_eval = self.t_total / (self.n_eval + 1 * 2)

                self._record_results(
                    Y=np.array([[now_metric]]),
                    macro_pos_all=np.array(now_macro_pos),
                    t_each_eval=t_each_eval,
                    avg_t_each_eval=avg_t_each_eval,
                )

                # update temperature                
                if self.n_eval % self.update_freq == 0:
                    self.T = self.decay * self.T

            # save checkpoint
            self._save_checkpoint(
                population=self.population,
                fitness=self.population_metric,
                temperature=self.T,
            )
    
    def _save_checkpoint(self, population, fitness, temperature):
        with open(os.path.join(self.checkpoint_path, "sa.pkl"), "wb") as f:
            pickle.dump(
                {
                    "population" : population,
                    "fitness" : fitness,
                    "temperature" : temperature
                },
                file=f
            )
        
        super()._save_checkpoint()
        
    
    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            super()._load_checkpoint()
            with open(os.path.join(self.args.checkpoint, "sa.pkl"), "rb") as f:
                checkpoint = pickle.load(f)
                self.start_from_checkpoint = True
        else:
                checkpoint = None
                self.start_from_checkpoint = False
        
        return checkpoint
            
    

        


            
