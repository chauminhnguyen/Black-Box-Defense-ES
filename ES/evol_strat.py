from abstracts import OOOptimizer
import numpy as np

class GES(OOOptimizer):
    def __init__(self, 
                 sigma         = 0.1,
                 alpha         = 0.5,
                 learning_rate = 0.01,
                 pop_size      = 256,
                 elite_size    = 256,
                 warm_up       = 20,
                 seed	       = 0):
        self.sigma         = sigma
        self.alpha         = alpha
        self.learning_rate = learning_rate
        assert(pop_size % 2 == 0), "Population size must be even"
        self.pop_size, self.half_popsize = pop_size, pop_size // 2
        self.elite_size    = elite_size
        self.warm_up       = warm_up
        self.rg 	= np.random.RandomState(seed)
    