import json
import os
import math
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
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from paths import get_output_directory, get_ft_output_directory

# Use our pretrained backbone
def main(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    n_episodes = 600
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot
    n_epoch = int( math.ceil(n_data / 4) * params.ft_epochs / math.ceil(n_data / bs) )

    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot
    use_fixed_features = params.ft_augmentation is None and params.ft_parts == 'head'

    # load support, support_clean, query samples
    support_epochs = 1 if use_fixed_features else n_epoch
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=params.ft_augmentation,
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
    
    query_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=False,
                                                   n_query_shot=q, n_episodes=n_episodes, augmentation=None,
                                                   unlabeled_ratio=params.unlabeled_ratio,
                                                   num_workers=params.num_workers,
                                                   split_seed=params.split_seed,
                                                   episode_seed=params.ft_episode_seed)
    query_iterator = iter(query_loader)
    support_iterator = iter(support_loader)

    # load pretrained backbone file
    backbone = get_backbone_class(params.backbone)() 
    body = get_model_class(params.model)(backbone, params)
    body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'
    state = torch.load(body_state_path)
    body.load_state_dict(copy.deepcopy(state))
    body.cuda()
    body.eval() # DO NOT UPDATE BODY!!
    

    # path configuration
    feature_path = './feature'
    feature_path = os.path.join(feature_path, params.target_dataset)
    feature_path = os.path.join(feature_path, '{:03d}way_{:03d}shot'.format(w, s))
    

    mix_bool = (params.ft_mixup or params.ft_cutmix or params.ft_manifold_mixup) 
    aug_bool = mix_bool or params.ft_augmentation
    
    if not aug_bool: 
        input_type = 'baseline'
    else:
        if params.ft_mixup:
            input_type ='mixup'
        elif params.ft_cutmix:
            input_type ='cutmix'
        elif params.ft_manifold_mixup:
            input_type ='manifold_mixup'
        elif params.ft_augmentation:
            input_type = params.ft_augmentation

    feature_path = os.path.join(feature_path, input_type)
    os.makedirs(feature_path, exist_ok=True)
    print(feature_path)
   
    # extract features
    for episode in tqdm(range(n_episodes)):
        with torch.no_grad():
            x_support, _ = next(support_iterator)
            x_query, _ = next(query_iterator)
            x_support = x_support.cuda()
            x_query = x_query.cuda()

            f_support = body.forward_features(x_support, params.ft_features)
            f_query = body.forward_features(x_query, params.ft_features)

        np.save(feature_path+'/support_{:03d}.npy'.format(episode), f_support.cpu().numpy())
        np.save(feature_path+'/query_{:03d}.npy'.format(episode), f_query.cpu().numpy())
    print("Saving Finished")


# Use pretrained backbone from Pytorch, load best & worst support/query sets
# def main2(params):
#     print(params.source_dataset)
#     n_episodes = 10
#     # body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'
#     # state = torch.load(body_state_path)
#     # source_dir = './logs/baseline/output/resnet10_simclr_LS_default/{}/05way_005shot_head_default'.format(params.source_dataset)

#     # backbone = get_backbone_class(params.backbone)()
#     # body = get_model_class(params.model)(backbone, params)
#     # body.load_state_dict(copy.deepcopy(state))

#     model = torchvision.models.resnet101(pretrained = True)

#     paths = []
#     # sucess 3, fail 3
#     for rank in range(1, 6):
#         best_path = os.path.join(source_dir, 'best{}.txt'.format(rank))
#         paths.append(best_path)
#     for rank in range(1, 6):
#         worst_path = os.path.join(source_dir, 'worst{}.txt'.format(rank))
#         paths.append(worst_path)

#     # same data
#     for episode in range(n_episodes):
#         # Reset models for each episode

#         support_feature_all = np.empty((0,1000))
#         query_feature_all = np.empty((0,1000))


#         with open(paths[episode], 'rb') as f:
#             test_acc_his = pickle.load(f)
#             x_support = pickle.load(f)
#             y_support = pickle.load(f)
#             x_query = pickle.load(f)
#             y_query = pickle.load(f)

#         for x in x_support:
#             with torch.no_grad():
#                 # x = x.cuda()
#                 # body를 통과한 extracted feature
#                 # f_support shape : (25, 512)
#                 # feature = body.forward_features(x.unsqueeze(0).cpu(), params.ft_features)
#                 feature = model(x.unsqueeze(0).cpu())
#                 support_feature_all = np.append(support_feature_all , feature, axis=0)

#         for x in x_query:
#             with torch.no_grad():
#                 # x = x.cuda()
#                 # body를 통과한 extracted feature
#                 # f_support shape : (25, 512)
#                 #feature = body.forward_features(x.unsqueeze(0).cpu(), params.ft_features)
#                 feature = model(x.unsqueeze(0).cpu())
#                 query_feature_all = np.append(query_feature_all , feature, axis=0)
#         path = './logs/img_ft_difference/feature/'+params.source_dataset+'/resnet101/5shot/{}_x_support.npy'.format(paths[episode][-9:-4])
#         np.save(path, support_feature_all) 
#         np.save('./logs/img_ft_difference/feature/'+params.source_dataset+'/resnet101/5shot/{}_x_query.npy'.format(paths[episode][-9:-4]), query_feature_all)
#         print("Saving Finished")



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
            main(params)
