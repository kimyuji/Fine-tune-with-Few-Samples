import copy
import json
import math
import os
import pickle
from this import d
import pandas as pd
import torch.nn as nn
import torchsummary as summary
import itertools
from backbone import get_backbone_class
import backbone
from datasets.dataloader import get_labeled_episodic_dataloader
from datasets.transforms import rand_bbox
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path
from utils import *
#from elastic_weight_consolidation import ElasticWeightConsolidation

from sklearn.cluster import KMeans 
from sklearn.metrics.cluster import v_measure_score

# output_dir : ./logs/output_bs3/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# base_output_dir : #./logs/output_baseline/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# 둘다 makedir true

def main(params):
    # ft_scheduler configuration 

    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    base_output_dir = get_output_directory(params) 
    output_dir = get_ft_output_directory(params)
    print()
    print('Running fine-tune with output folder:')
    print(output_dir)
    print()

    # Settings
    n_episodes = 600
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot
    # if params.ft_with_clean:
    #     n_data = n_data*2
    n_epoch = int( math.ceil(n_data / 4) * params.ft_epochs / math.ceil(n_data / bs) )
    print()

    # Model
    backbone = get_backbone_class(params.backbone)() 
    backbone_pre = get_backbone_class(params.backbone)() 
    body = get_model_class(params.model)(backbone, params)
    pretrained = get_model_class(params.model)(backbone_pre, params)

    if params.ft_features is not None:
        if params.ft_features not in body.supported_feature_selectors:
            raise ValueError(
                'Feature selector "{}" is not supported for model "{}"'.format(params.ft_features, params.model))


    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)

    layer_path = train_history_path.replace('train_history', 'layer_diff')

    params_path = get_ft_params_path(output_dir)

    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    #print('Saving finetune validation history to {}'.format(train_history_path))
    print()
    
    # 저장할 dataframe
    df_layer = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['layer{}'.format(e + 1) for e in range(42)])

    # Pre-train state
    pretrain_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'

    print('Using pre-train state:')
    print(pretrain_state_path)
    print()
    state = torch.load(pretrain_state_path)
    pretrained.load_state_dict(copy.deepcopy(state))
    pretrained.eval()
########################################################################################################################
    for episode in range(n_episodes):
        body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/mini_test/05way_001shot_full_default/body_{:03d}.pt'.format(episode+1)
        body.load_state_dict(torch.load(body_state_path))
        body.eval()
        
        layer_diff = []
        for p, b in zip(pretrained.named_parameters(), body.named_parameters()):
            layer_diff.append(np.abs((p[1]-b[1]).detach().numpy()).mean())
            #print(p_param.shape, b_param.shape)

        # opt_params = []
        # if params.ft_train_body:
        #     opt_params.append({'params': body.parameters()})

        df_layer.loc[episode+1] = layer_diff

    print('Saved history to:')
    print(layer_path)
    df_layer.to_csv(layer_path)


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    targets = params.target_dataset
    if targets is None:
        targets = [targets]
    elif len(targets) > 1:
        print('#' * 80)
        print("Running finetune iteratively for multiple target datasets: {}".format(targets))
        print('#' * 80)

    for target in targets:
        params.target_dataset = target
        main(params)
