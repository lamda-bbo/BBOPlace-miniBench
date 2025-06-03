import os
import numpy as np
import logging
import pickle
from abc import abstractmethod
from utils.debug import *
from utils.constant import INF
from utils.random_parser import set_state
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class BasicAlgo:
    def __init__(self, args, evaluator, logger) -> None:
        self.args = args
        self.eval_metrics = args.eval_metrics
        self.evaluator = evaluator
        self.logger = logger

        self.n_eval = 0
        self.population = None
        self.best_Y = np.zeros(len(self.eval_metrics)) + INF 

        self.nds = NonDominatedSorting()
        self.pareto_front = None

        self.t_total = 0
        self.max_eval_time_second = args.max_eval_time * 60 * 60 

        self.checkpoint_path = os.path.join(args.result_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    
    @abstractmethod
    def run(self):
        pass

    def _record_results(self, Y, macro_pos_all, t_each_eval=0, avg_t_each_eval=0):
        pop_best_Y = np.min(Y, axis=0)
        pop_avg_Y  = np.mean(Y, axis=0)
        pop_std_Y  = np.std(Y, axis=0)

        # pareto sort
        combined_Y = Y if self.pareto_front is None else np.row_stack([Y, self.pareto_front])
        pareto_front_indices = np.array(self.nds.do(combined_Y)[0])

        self.pareto_front = combined_Y[pareto_front_indices]
        selected_indices = pareto_front_indices[pareto_front_indices < Y.shape[0]]

        for idx, (y, m_pos) in enumerate(zip(Y, macro_pos_all)):
            self.n_eval += 1
            if self.n_eval > self.args.max_evals:
                break

            self.best_Y = np.minimum(self.best_Y, y)
            
            if idx in selected_indices:
                y_info = "\t".join(
                    [f"{key}: {value}" for key, value in zip(self.eval_metrics, y)]
                )
                logging.info(f"n_eval: {self.n_eval}\t" + y_info)

                if len(m_pos) > 0:
                    self.evaluator.placer.save_placement(
                        macro_pos=m_pos,
                        n_eval=self.n_eval,
                    )
                    self.evaluator.placer.plot(
                        macro_pos=m_pos,
                        n_eval=self.n_eval,
                    )

            for idx, metric in enumerate(self.eval_metrics):
                self.logger.add(f"{metric}/current", y[idx])
                self.logger.add(f"{metric}/his_best", self.best_Y[idx])
                self.logger.add(f"{metric}/pop_best", pop_best_Y[idx])
                self.logger.add(f"{metric}/pop_avg",  pop_avg_Y[idx])
                self.logger.add(f"{metric}/pop_std",  pop_std_Y[idx])
            
            self.logger.add("Time/each_eval", t_each_eval)
            self.logger.add("Time/avg_each_eval", avg_t_each_eval)
            self.logger.step()

            self.evaluator.placer.save_metrics(
                n_eval=self.n_eval,
                current_Y=y,
                his_best_Y=self.best_Y,
                pop_best_Y=pop_best_Y,
                pop_avg_Y=pop_avg_Y,
                pop_std_Y=pop_std_Y,
                t_each_eval=t_each_eval,
                avg_t_each_eval=avg_t_each_eval
            )
        

    def _save_checkpoint(self):
        logging.info("saving checkpoint")

        # logger checkpoint
        self.logger._save_checkpoint(path=self.checkpoint_path)

        # placement and corresponding figure checkpoint
        self.evaluator.placer._save_checkpoint(checkpoint_path=self.checkpoint_path)

        # saving pareto front
        np.save(os.path.join(self.checkpoint_path, "pareto_front.npy"), self.pareto_front)

        if self.t_total >= self.max_eval_time_second:
            logging.info(f"Reaching maximun running time ({self.t_total:.2f} >= {self.max_eval_time_second}), the program will exit")
            exit(0)

        
    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            logging.info(f"Loading checkpoint from {self.args.checkpoint}")
            log_file = os.path.join(self.args.checkpoint, "log.pkl")
            with open(log_file, 'rb') as log_f:
                log_data = pickle.load(log_f)
            
            
            self.n_eval = len(log_data[f"{self.eval_metrics[0]}/current"])
            
            set_state(log_data)

            for i_eval in range(0, self.n_eval):
                for key, value_lst in log_data.items():
                    if key in ("random", "np_random"):
                        continue
                    self.logger.add(key, value_lst[i_eval])
                self.logger.step()

                self.evaluator.placer.save_metrics(
                    n_eval=i_eval+1,
                    current_Y=np.array([log_data[f"{metric}/current"][i_eval] for metric in self.eval_metrics]),
                    his_best_Y=np.array([log_data[f"{metric}/his_best"][i_eval] for metric in self.eval_metrics]), 
                    pop_best_Y=np.array([log_data[f"{metric}/pop_best"][i_eval] for metric in self.eval_metrics]), 
                    pop_avg_Y=np.array([log_data[f"{metric}/pop_avg"][i_eval] for metric in self.eval_metrics]), 
                    pop_std_Y=np.array([log_data[f"{metric}/pop_std"][i_eval] for metric in self.eval_metrics]),
                    t_each_eval=log_data["Time/each_eval"][i_eval],
                    avg_t_each_eval=log_data["Time/avg_each_eval"][i_eval]
                )

            self.best_Y = np.array([log_data[f"{metric}/his_best"][self.n_eval-1] for metric in self.eval_metrics])
            self.t_total   = sum(log_data["Time/each_eval"])

            self.evaluator.placer._load_checkpoint(checkpoint_path=self.args.checkpoint)


            # pareto front
            self.pareto_front = np.load(os.path.join(self.args.checkpoint, "pareto_front.npy"))