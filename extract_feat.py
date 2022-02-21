import json
import os

import copy
from tkinter import X
import numpy as np
import pandas as pd
import torch
import torch.optim
import torchvision
from tqdm import tqdm
import pickle

from backbone import get_backbone_class
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from io_utils import parse_args
from model import get_model_class
from paths import get_output_directory, get_ft_output_directory


def _get_dataloaders(params):
    batch_size = 1
    labeled_source_bs = batch_size 
    unlabeled_source_bs = batch_size
    unlabeled_target_bs = batch_size

    if params.us and params.ut:
        unlabeled_source_bs //= 2
        unlabeled_target_bs //= 2

    ls, us, ut = None, None, None  # labeled source, unlabeled source, unlabeled target

    print('Using source data {} (labeled)'.format(params.source_dataset))
    ls = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                        batch_size=labeled_source_bs, num_workers=params.num_workers)


    return ls, us, ut


def main(params):
    print(params.source_dataset)
    base_output_dir = get_output_directory(params) 
    body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'
    state = torch.load(body_state_path)
    
    

    backbone = get_backbone_class(params.backbone)()
    body = get_model_class(params.model)(backbone, params)
    body.load_state_dict(copy.deepcopy(state))

    output_dir = get_output_directory(params)
    labeled_source_loader, unlabeled_source_loader, unlabeled_target_loader = _get_dataloaders(params)

    model = torchvision.models.resnet101(pretrained = True)
    
    feature_all = np.empty((0,1000))
    label_all = np.array([[]])
    for x, y in tqdm(labeled_source_loader):
        with torch.no_grad():
            # x = x.cuda()
            # body를 통과한 extracted feature
            # f_support shape : (25, 512)
            # feature = body.forward_features(x, params.ft_features)
            feature = model(x)
            feature_all = np.append(feature_all , feature, axis=0)
            label_all = np.append(label_all, y)

    np.save('./feature/'+params.source_dataset+'_feature.npy', feature_all)
    np.save('./feature/'+params.source_dataset+'_label.npy', label_all)
    print("Saving Finished")

def main2(params):
    print(params.source_dataset)
    n_episodes = 10
    # body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'
    # state = torch.load(body_state_path)
    # source_dir = './logs/baseline/output/resnet10_simclr_LS_default/{}/05way_005shot_head_default'.format(params.source_dataset)

    # backbone = get_backbone_class(params.backbone)()
    # body = get_model_class(params.model)(backbone, params)
    # body.load_state_dict(copy.deepcopy(state))

    model = torchvision.models.resnet101(pretrained = True)

    paths = []
    # sucess 3, fail 3
    for rank in range(1, 6):
        best_path = os.path.join(source_dir, 'best{}.txt'.format(rank))
        paths.append(best_path)
    for rank in range(1, 6):
        worst_path = os.path.join(source_dir, 'worst{}.txt'.format(rank))
        paths.append(worst_path)

    # same data
    for episode in range(n_episodes):
        # Reset models for each episode

        support_feature_all = np.empty((0,1000))
        query_feature_all = np.empty((0,1000))


        with open(paths[episode], 'rb') as f:
            test_acc_his = pickle.load(f)
            x_support = pickle.load(f)
            y_support = pickle.load(f)
            x_query = pickle.load(f)
            y_query = pickle.load(f)

        for x in x_support:
            with torch.no_grad():
                # x = x.cuda()
                # body를 통과한 extracted feature
                # f_support shape : (25, 512)
                # feature = body.forward_features(x.unsqueeze(0).cpu(), params.ft_features)
                feature = model(x.unsqueeze(0).cpu())
                support_feature_all = np.append(support_feature_all , feature, axis=0)

        for x in x_query:
            with torch.no_grad():
                # x = x.cuda()
                # body를 통과한 extracted feature
                # f_support shape : (25, 512)
                #feature = body.forward_features(x.unsqueeze(0).cpu(), params.ft_features)
                feature = model(x.unsqueeze(0).cpu())
                query_feature_all = np.append(query_feature_all , feature, axis=0)
        path = './logs/img_ft_difference/feature/'+params.source_dataset+'/resnet101/5shot/{}_x_support.npy'.format(paths[episode][-9:-4])
        np.save(path, support_feature_all) 
        np.save('./logs/img_ft_difference/feature/'+params.source_dataset+'/resnet101/5shot/{}_x_query.npy'.format(paths[episode][-9:-4]), query_feature_all)
        print("Saving Finished")



if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('pretrain')
    # ['miniImageNet", "miniImageNet_test", "CropDisease", "EuroSAT", "ISIC", "ChestX"]
    # datas = ["miniImageNet"]
    datas = ["mini_test", "crop", "euro", "isic", "chest"]
    for data in datas:
        params.source_dataset = data
        params.backbone = 'resnet10'
        params.model = 'simclr'

        targets = params.target_dataset
        if targets is None:
            targets = [targets]
        elif len(targets) > 1:
            print('#' * 80)
            print("Running pretrain iteratively for multiple target datasets: {}".format(targets))
            print('#' * 80)

        for target in targets:
            params.target_dataset = target
            main2(params)
