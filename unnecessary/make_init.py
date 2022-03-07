import copy
import json
import math
import os
import pickle
import pandas as pd
import torch.nn as nn

from backbone import get_backbone_class
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path
from utils import *

params = parse_args('train')
params.model = 'simclr'
params.n_way = 5
n_inits = 600
backbone = get_backbone_class(params.backbone)()
body = get_model_class(params.model)(backbone, params)
for i in range(n_inits):
	head = get_classifier_head_class(params.ft_head)(body.final_feat_dim, params.n_way, params)
	torch.save(head.state_dict(), './classifier_init/random_init_{:03d}.pt'.format(i))
print("Saving Finished")