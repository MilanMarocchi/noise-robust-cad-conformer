import random

import numpy as np
import torch

# Set seed to random for the equivalent of having no seed.
MAKE_SEEDED = True 
SEED = 0x5EED if MAKE_SEEDED else random.randint(0, 2**32 - 1) # ayy

# NOTE: Making entirely reproducible severly slows computations
#       this current seeding gives good enough reproducibility 
# torch.use_deterministic_algorithms(True)
MAKE_REPRODUCIBLE = False

if MAKE_REPRODUCIBLE:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
else:
    torch.backends.cudnn.benchmark = True

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


torch_gen = torch.Generator()
torch_gen.manual_seed(SEED)
