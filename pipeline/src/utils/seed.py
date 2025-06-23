import os
import random
import numpy as np
import torch

# Fixed seeds for reproducibility
GLOBAL_SEED = 42


def set_seeds(seed=GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment variable for Python hash
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"All seeds set to: {seed}")


# Automatically set seeds when this module is imported
set_seeds()

# Export the function for manual use if needed
__all__ = ["set_seeds", "GLOBAL_SEED"]
