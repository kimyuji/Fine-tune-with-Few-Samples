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
    torch_pretrained = ("torch" in params.backbone)
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
    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot
    # Whether to optimize for fixed features (when there is no augmentation and only head is updated)
    use_fixed_features = params.ft_augmentation is None and params.ft_parts == 'head'

    # Model
    backbone = get_backbone_class(params.backbone)() 
    body = get_model_class(params.model)(backbone, params)
    pretrained = get_model_class(params.model)(backbone, params)

    if params.ft_features is not None:
        if params.ft_features not in body.supported_feature_selectors:
            raise ValueError(
                'Feature selector "{}" is not supported for model "{}"'.format(params.ft_features, params.model))

    # Dataloaders
    # Note that both dataloaders sample identical episodes, via episode_seed
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
    if params.ft_clean_test and not use_fixed_features:
        support_loader_clean = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=None,
                                                     unlabeled_ratio=params.unlabeled_ratio,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed, episode_seed=params.ft_episode_seed)
        support_iterator_clean = iter(support_loader_clean)
    #print("dddd")
    # 값이 맞게끔 보증! 
    assert (len(query_loader) == n_episodes)
    assert (len(support_loader) == n_episodes * support_epochs)

    query_iterator = iter(query_loader)
    support_iterator = iter(support_loader)
    support_batches = math.ceil(n_data / bs)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    
    if params.ft_manifold_aug:
        train_history_path = train_history_path.replace('.csv', '_{}.csv'.format(params.ft_manifold_aug))
        test_history_path = test_history_path.replace('.csv', '_{}.csv'.format(params.ft_manifold_aug))
    if params.ft_label_smoothing != 0:
        train_history_path = train_history_path.replace('.csv', '_ls.csv')
        test_history_path = test_history_path.replace('.csv', '_ls.csv')

    if params.ft_scheduler_start != params.ft_scheduler_end:
        train_history_path = train_history_path.replace('.csv', '_{}_{}.csv'.format(params.ft_scheduler_start, params.ft_scheduler_end))
        test_history_path = test_history_path.replace('.csv', '_{}_{}.csv'.format(params.ft_scheduler_start, params.ft_scheduler_end))

    loss_history_path = train_history_path.replace('train_history', 'loss_history')
    train_clean_history_path = test_history_path.replace('test_history', 'clean_history')
    support_v_score_history_path = train_history_path.replace('train_history', 'v_score_support')
    query_v_score_history_path = train_history_path.replace('train_history', 'v_score_query')

    params_path = get_ft_params_path(output_dir)

    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    #print('Saving finetune validation history to {}'.format(train_history_path))
    print()
    # saving parameters on this json file
    with open(params_path, 'w') as f_batch:
        json.dump(vars(params), f_batch, indent=4)
    
    # 저장할 dataframe
    df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_train_clean = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    # df_grad = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
    #                        columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_v_score_query = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch0'])

    # Pre-train state
    
    if not torch_pretrained : 
        if params.ft_pretrain_epoch is None: # best state
            body_state_path = get_final_pretrain_state_path(base_output_dir)
        else: # 원하는 epoch수의 state를 받아오고 싶다면 
            body_state_path = get_pretrain_state_path(base_output_dir, params.ft_pretrain_epoch)
            
        if not os.path.exists(body_state_path):
            raise ValueError('Invalid pre-train state path: ' + body_state_path)
            
        print('Using pre-train state:')
        print(body_state_path)
        print()
        state = torch.load(body_state_path)


    all_cases = list(itertools.permutations(list(range(w))))
    class_shuffled = all_cases
    for case in copy.deepcopy(all_cases):
        for idx in range(w):
            if case[idx] == idx : 
                class_shuffled.remove(case)
                break 
    pretrained.load_state_dict(copy.deepcopy(state))
########################################################################################################################
    for episode in range(n_episodes):
        # Reset models for each episode
        if params.ft_no_pretrain:
            backbone = get_backbone_class(params.backbone)() 
            body = get_model_class(params.model)(backbone, params)
        else:
            if not torch_pretrained:
                body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.

        head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features

        body.cuda()
        head.cuda()

        opt_params = []
        if params.ft_train_head:
            opt_params.append({'params': head.parameters()})
        if params.ft_train_body:
            opt_params.append({'params': body.parameters()})

        # Optimizer and Learning Rate Scheduler
        # select optimizer
        if params.ft_optimizer == 'SGD':
            optimizer = torch.optim.SGD(opt_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        elif params.ft_optimizer == 'Adam':
            optimizer = torch.optim.Adam(opt_params, lr=params.ft_lr, weight_decay=0.001)
        elif params.ft_optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(opt_params, lr=params.ft_lr, momentum=0.9, weight_decay=0.001)
        elif params.ft_optimizer == 'Adagrad':
            optimizer = torch.optim.Adagrad(opt_params, lr=params.ft_lr, weight_decay=0.001)
        elif params.ft_optimizer == 'RMSprop_no_momentum':
            optimizer = torch.optim.RMSprop(opt_params, lr=params.ft_lr, weight_decay=0.001)

        # select learning rate scheduler
        if params.ft_lr_scheduler == "CosAnneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # T_max : 최대 iteration 횟수
        elif params.ft_lr_scheduler == "CosAnneal_WS":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        elif params.ft_lr_scheduler == "Cycle":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, step_size_up=5, max_lr=0.03, gamma=0.5, mode='exp_range')
        elif params.ft_lr_scheduler == 'Exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

        # Loss function
        #criterion = nn.CrossEntropyLoss(label_smoothing=params.ft_label_smoothing).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        #ewc = ElasticWeightConsolidation(head, criterion, optimizer)

        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]

        x_query = next(query_iterator)[0].cuda()
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).cuda() 


        # Evaluation
        body.eval()
        head.eval()

        scores = []
        #if params.layer_diff:
        # 7 layers
        # for name, param in body.named_parameters():
        #     print(name)
        #     print(param.shape)

        # Test Using Query
        with torch.no_grad():
            if not use_fixed_features:
                if not torch_pretrained:
                    f_query = body.forward_features(x_query, params.ft_features)
                else:
                    f_query = backbone(x_query)
                    f_query = f_query.squeeze(-1).squeeze(-1)
            p_query = head(f_query) 
        test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
        f_query = f_query.cpu().numpy()
        cluster_label = y_query.cpu().numpy()
        kmeans = KMeans(n_clusters = w)
        cluster_pred = kmeans.fit(f_query).labels_
        score = v_measure_score(cluster_pred, cluster_label)
        print(score)
        df_v_score_query.loc[episode + 1] = score
    

    df_v_score_query.to_csv(query_v_score_history_path + '_0')
    
    


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
