import numpy as np
import random

import torch


# set random seed for all possible sources of randomness
def set_random_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# two-sided p-value based on a (hypothetically) symmetric empirical null distirubtion
def pvalue(empirical, score):
    p1 = np.mean(empirical <= score)
    p2 = np.mean(empirical >= score)
    return 2 * min(p1, p2)
