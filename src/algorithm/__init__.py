REGISTRY = {}

from .ea.ea import EA
from .bo.bo import BO 
from .ea.sa import SA
from .ea.nsgaii import NSGAII
from .cma_es.cma_es import CMAES
from .pso.pso import PSO

REGISTRY["ea"] = EA
REGISTRY["bo"] = BO
REGISTRY["sa"] = SA
REGISTRY["nsgaii"] = NSGAII
REGISTRY["cma_es"] = CMAES
REGISTRY["pso"] = PSO