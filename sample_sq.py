import copy
import json
import math
import os
import pickle
import pandas as pd
import numpy as np
from sympy import S
import torch.nn as nn
import torchsummary as summary
import torchvision
from tqdm import tqdm

from backbone import get_backbone_class
import backbone
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path
from utils import *

def main(params):
    base_output_dir = get_output_directory(params) 
    output_dir = get_ft_output_directory(params)
    torch_pretrained = ("torch" in params.backbone)
    print()
    print('Running fine-tune with output folder:')
    print(output_dir)
    print()

    n_set = 20

    if params.n_shot == 1:
        total_shot = 16 # 1(support shot) + 15(query shot) 
    else:
        total_shot = 20
    s = total_shot * n_set 

    q = 0
    w = params.n_way

    # Settings
    n_episodes = 1 # to avoid overlapping, we sample all the imgs in 1 episode

    print()

    # Dataloaders
    # Note that both dataloaders sample identical episodes, via episode_seed
    support_epochs = 1 
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=params.ft_augmentation,
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
    
    support_iterator = iter(support_loader)

    # use pretrained resnet 152 to extract features for analysis
    feat_extract = torchvision.models.resnet152(pretrained = True)

    # use pretrained resnet 18 to extract features for evaluating
    backbone = get_backbone_class(params.backbone)()
    

    with torch.no_grad():
        x_support, _ = next(support_iterator)

    # feature extraction 
    # for x in tqdm(x_support):
    #     with torch.no_grad():
    #         feature = feat_extract(x.unsqueeze(0).cpu())
    #         #feature = model.forward_features(x.unsqueeze(0).cpu(), params.ft_features)
    #         support_feature_all = np.append(support_feature_all , feature, axis=0)

    #np.random.shuffle(support_feature_all) # random shuffling

    # # divide 5 classes, each shaped (params.n_shot, 1000)
    class0 = np.random.permutation(x_support[ 0 : s])
    class1 = np.random.permutation(x_support[ s : 2*s])
    class2 = np.random.permutation(x_support[ 2*s : 3*s])
    class3 = np.random.permutation(x_support[ 3*s : 4*s])
    class4 = np.random.permutation(x_support[ 4*s : 5*s])

    for i in tqdm(range(n_set)):
        class0_ep = class0[i * total_shot : (i+1) * total_shot]
        class1_ep = class1[i * total_shot : (i+1) * total_shot]
        class2_ep = class2[i * total_shot : (i+1) * total_shot]
        class3_ep = class3[i * total_shot : (i+1) * total_shot]
        class4_ep = class4[i * total_shot : (i+1) * total_shot]
        class_all_ep = np.stack([class0_ep, class1_ep, class2_ep, class3_ep, class4_ep], axis=0)

        sampled_support = class_all_ep[: , :params.n_shot].reshape(-1, 3, 224, 224)
        sampled_query = class_all_ep[: , params.n_shot:].reshape(-1, 3, 224, 224)

        sampled_support = torch.Tensor(sampled_support).cpu()
        sampled_query = torch.Tensor(sampled_query).cpu()
        
        # extract features
        feat_sampled_support = feat_extract(sampled_support).detach().numpy()
        feat_sampled_query = feat_extract(sampled_query).detach().numpy()
        embed_sampled_support = backbone(sampled_support).detach().numpy()
        embed_sampled_query = backbone(sampled_query).detach().numpy()
        
        feature_path = os.path.join(output_dir, "feature")
        embedding_path = os.path.join(output_dir, "embedding")

        os.makedirs(feature_path, exist_ok=True)
        os.makedirs(embedding_path, exist_ok=True)

        feat_path_support = os.path.join(feature_path, "{:03d}_support.npy".format(i))
        feat_path_query = os.path.join(feature_path, "{:03d}_query.npy".format(i))
        embed_path_support = os.path.join(embedding_path, "{:03d}_support.npy".format(i))
        embed_path_query = os.path.join(embedding_path, "{:03d}_query.npy".format(i))

        np.save(feat_path_support, feat_sampled_support) 
        np.save(feat_path_query, feat_sampled_query) 

        np.save(embed_path_support, embed_sampled_support) 
        np.save(embed_path_query, embed_sampled_query) 

    print("Saving ended!") 

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
