from .mutation import GGShuffleMutation, SPInversionMutation
from .crossover import GGUniformCrossover, SPOrderCrossover

REGISTRY = {}

REGISTRY["mutation"] = {
    "gg" : GGShuffleMutation,
    "sp" : SPInversionMutation
}

REGISTRY["crossover"] = {
    "gg" : GGUniformCrossover,
    "sp" : SPOrderCrossover
}

