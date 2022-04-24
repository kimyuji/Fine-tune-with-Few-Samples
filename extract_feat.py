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
from datasets.transforms import rand_bbox
from backbone import get_backbone_class
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from paths import get_output_directory, get_ft_output_directory
from torchvision import transforms


# Use our pretrained backbone
def main(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    n_episodes = 1
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot
    n_epoch = int( math.ceil(n_data / 4) * params.ft_epochs / math.ceil(n_data / bs) )

    w = params.n_way
    s = 20
    q = 20

    # load clean support and query
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=1,
                                                     augmentation=None,
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
    query_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=False,
                                                   n_query_shot=q, n_episodes=n_episodes, augmentation=None,
                                                   unlabeled_ratio=params.unlabeled_ratio,
                                                   num_workers=params.num_workers,
                                                   split_seed=params.split_seed,
                                                   episode_seed=params.ft_episode_seed)

    # augmentation
    support_loader_flip = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=n_epoch,
                                                     augmentation='randomhorizontalflip',
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
    support_loader_crop = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=n_epoch,
                                                     augmentation='randomresizedcrop',
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)    
    # clean
    x_support, _ = next(iter(support_loader))
    x_query, _ = next(iter(query_loader))
    x_support = x_support.cuda()
    x_query = x_query.cuda()
    # aug
    x_support_flip, _ = next(iter(support_loader_flip))
    x_support_crop, _ = next(iter(support_loader_crop))
    x_support_flip = x_support_flip.cuda()
    x_support_crop = x_support_crop.cuda()
    # mix
    #lam = np.random.beta(1.0, 1.0)
    lam = 0.5
    bbx1, bby1, bbx2, bby2 = rand_bbox(x_support.shape, lam)
    indices_shuffled = torch.randperm(x_support.shape[0])
    x_support_cutmix = copy.deepcopy(x_support)
    x_support_cutmix[:,:,bbx1:bbx2, bby1:bby2] = x_support[indices_shuffled,:,bbx1:bbx2, bby1:bby2]
    x_support_mixup = lam * x_support[:,:,:] + (1. - lam) * x_support[indices_shuffled,:,:]

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
   
    # extract features
    with torch.no_grad():
        # # clean
        # f_support = body.forward_features(x_support, params.ft_features)
        # f_query = body.forward_features(x_query, params.ft_features)
        # aug
        f_flip = body.forward_features(transforms.functional.hflip(x_support), params.ft_features)
        f_crop = body.forward_features(x_support_crop, params.ft_features)
        # # mix
        # f_cutmix = body.forward_features(x_support_cutmix, params.ft_features)
        # f_mixup = body.forward_features(x_support_mixup, params.ft_features)
        # # manifold mixup
        # f_manifold_mixup = lam * f_support[:,:] + (1. - lam) * f_support[indices_shuffled][:,:]

    # # clean support, query
    # feature_path_final = os.path.join(feature_path, 'baseline')
    # os.makedirs(feature_path_final, exist_ok=True)
    # np.save(feature_path_final+'/support.npy', f_support.cpu().numpy())
    # np.save(feature_path_final+'/query.npy', f_query.cpu().numpy())

    # # mix
    # feature_path_final = os.path.join(feature_path, 'cutmix')
    # os.makedirs(feature_path_final, exist_ok=True)
    # np.save(feature_path_final+'/support.npy', f_cutmix.cpu().numpy())

    # feature_path_final = os.path.join(feature_path, 'mixup')
    # os.makedirs(feature_path_final, exist_ok=True)
    # np.save(feature_path_final+'/support.npy', f_mixup.cpu().numpy())

    # feature_path_final = os.path.join(feature_path, 'manifold_mixup')
    # os.makedirs(feature_path_final, exist_ok=True)
    # np.save(feature_path_final+'/support.npy', f_manifold_mixup.cpu().numpy())

    # aug
    feature_path_final = os.path.join(feature_path, 'flip')
    os.makedirs(feature_path_final, exist_ok=True)
    np.save(feature_path_final+'/support.npy', f_flip.cpu().numpy())

    feature_path_final = os.path.join(feature_path, 'crop')
    os.makedirs(feature_path_final, exist_ok=True)
    np.save(feature_path_final+'/support.npy', f_crop.cpu().numpy())

    print("Saving Finished")



def extract_source(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    backbone = get_backbone_class(params.backbone)() 
    body = get_model_class(params.model)(backbone, params)
    body_state_path = './logs/baseline/output/resnet10_simclr_LS_default/pretrain_state_1000.pt'
    state = torch.load(body_state_path)
    body.load_state_dict(copy.deepcopy(state))
    body.cuda()
    body.eval()

    labeled_source_bs = 1
    labeled_source_loader = get_dataloader(dataset_name='miniImageNet', augmentation='strong',
                        batch_size=labeled_source_bs, num_workers=2)
    df_source = pd.DataFrame(None, index = list(range(64*20)), columns=list(range(512)))
    sample_count = [0]*64
    with torch.no_grad():
        for img, label in labeled_source_loader:
            clss = int(label[0])
            img = img.cuda()
            if sample_count[clss] >= 20 : continue
            feat = body.forward_features(img, params.ft_features)
            index = sample_count[clss] + 20 * clss
            df_source.loc[index] = feat.cpu().numpy()
            sample_count[clss] = int(sample_count[clss]) + 1
            if sum(sample_count) > 64*20+1: break
        source_path = './feature/miniImageNet/baseline/'
        os.makedirs(source_path, exist_ok=True)
        df_source.to_csv(source_path+'source_all_df.csv')



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
    params = parse_args('train')
    # ['miniImageNet", "miniImageNet_test", "CropDisease", "EuroSAT", "ISIC", "ChestX"]
    # datas = ["miniImageNet"]
    # datas = ["mini_test", "crop", "euro", "isic", "chest"]
    # for data in datas:
    #     params.source_dataset = data
    #     params.backbone = 'resnet10'
    #     params.model = 'simclr'

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
