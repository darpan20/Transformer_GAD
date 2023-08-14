# Standard libraries
import math
import os
from random import random

from functools import partial


# Plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
# PyTorch Lightning
import pytorch_lightning as pl



from tqdm import tqdm as tqdm

from pytorch_lightning import loggers as pl_loggers


# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

from tqdm.notebook import tqdm

DATASET_PATH = os.environ.get("datasets", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("model_checkpoints", "saved_models/Transformers/")

CHECKPOINT_PATH_RNN = os.environ.get("model_checkpoints", "saved_models/GRU_RNN/")

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)




import random

def _set_seeds(seed):


    """ sets all the required seed values for a run"""
        

    random.seed(seed)
    np.random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception as e:
        pass    
    try:
        import torch
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
    except Exception as e:
        pass
