from pymoo.core.problem import Problem
import numpy as np 

class PlacementProblem(Problem):
    def __init__(self, args, evaluator):
        self.eval_metrics = args.eval_metrics

        self.evaluator = evaluator
        n_var = evaluator.n_dim
        xl = evaluator.xl
        xu = evaluator.xu
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=len(self.eval_metrics),
            vtype=np.int64
        )

    def _evaluate(self, x, out, *args, **kwargs):
        y, macro_pos = self.evaluator.evaluate(x)

        if len(self.eval_metrics) == 1:
            out["F"] = y[self.eval_metrics[0]]
        else:
            out["F"] = np.column_stack(
                [y[metric] for metric in self.eval_metrics]
            )
        out["macro_pos"] = macro_pos
