from pymoo.core.problem import Problem
import numpy as np 

class PlacementProblem(Problem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        n_var = evaluator.n_dim
        xl = evaluator.xl
        xu = evaluator.xu
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=1,
            vtype=np.int64
        )

    def _evaluate(self, x, out, *args, **kwargs):
        y, macro_pos = self.evaluator.evaluate(x)

        out["F"] = y
        out["macro_pos"] = macro_pos