'''
Author: fuchy@stu.pku.edu.cn
LastEditors: Please set LastEditors
Description: Network parameters and helper functions
FilePath: /compression/networkTool.py
'''

import torch
import os, random
import numpy as np

# Set which GPU to use (here, GPU 0). If CUDA is not available, use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters
bptt = 1024  # Context window length for training (used in models like RNNs)
expName = './Exp/Obj'  # Directory for experiment outputs
DataRoot = './Data/Obj'  # Directory for input data

checkpointPath = expName + '/checkpoint'  # Path for model checkpoints
levelNumK = 4  # Some model-specific level parameter

# Path to training data (expects .mat files). Note: run ImageFolder.calcdataLenPerFile() before using
trainDataRoot = DataRoot + "/train/*.mat"

# Description of the experiment
expComment = 'OctAttention, trained on MPEG 8i,MVUB 1~10 level. 2021/12. All rights reserved.'

MAX_OCTREE_LEVEL = 12  # Max depth of octree structure used (e.g. in point cloud compression)

# Set random seed for reproducibility
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False  # Disable autotuner for reproducibility
torch.backends.cudnn.deterministic = True  # Ensure reproducible results

# Prevent H5PY from modifying files (if using .h5 or .mat files)
os.environ["H5PY_DEFAULT_READONLY"] = "1"

# Save a model checkpoint
def save(index, saveDict, modelDir='checkpoint', pthType='epoch'):
    if os.path.dirname(modelDir) != '' and not os.path.exists(os.path.dirname(modelDir)):
        os.makedirs(os.path.dirname(modelDir))
    torch.save(saveDict, modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, index))

# Load a model checkpoint (handles both single and multi-GPU checkpoints)
def reload(checkpoint, modelDir='checkpoint', pthType='epoch', print=print, multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, checkpoint), map_location=device)
            pth = modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir, map_location=device)
            pth = modelDir
        saveDict['path'] = pth

        # If model was trained with DataParallel, remove 'module.' prefix from keys
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove 'module.'
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        return saveDict
    except Exception as e:
        print('**warning**', e, ' start from initial model')
    return None

# Logging utility: writes to console and a file
class CPrintl():
    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))

    def __call__(self, *args):
        print(*args)
        print(*args, file=open(self.log_file, 'a'))

# Utility to print model structure and parameter count
def model_structure(model, print=print):
    print('-' * 120)
    print('|' + ' ' * 30 + 'weight name' + ' ' * 31 + '|' \
          + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 120)
    num_para = 0
    for _, (key, w_variable) in enumerate(model.named_parameters()):
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape), each_para))
    print('-' * 120)
    print('The total number of parameters: ' + str(num_para))
    print('-' * 120)
.
