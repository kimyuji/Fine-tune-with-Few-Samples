import copy
import json
import math
import os
import pickle
import pandas as pd
import torch.nn as nn
import torchsummary as summary

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
from elastic_weight_consolidation import ElasticWeightConsolidation

# output_dir : ./logs/output_bs3/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# base_output_dir : #./logs/output_baseline/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# 둘다 makedir true

def main(params):
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
    n_epoch = int( math.ceil(n_data / 4) * params.ft_epochs / math.ceil(n_data / bs) )
    # print("\nCurrent batch size:", bs)
    # print("Current optimizer:", params.ft_optimizer)
    # print("Current learning rate:", params.ft_lr)
    # print("Currently updating :", params.ft_parts)
    # print("Current scheduler :", params.ft_lr_scheduler)
    print()
    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot
    # Whether to optimize for fixed features (when there is no augmentation and only head is updated)
    use_fixed_features = params.ft_augmentation is None and params.ft_parts == 'head'

    # Model
    backbone = get_backbone_class(params.backbone)()
    body = get_model_class(params.model)(backbone, params)
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

    # 값이 맞게끔 보증! 
    assert (len(query_loader) == n_episodes)
    assert (len(support_loader) == n_episodes * support_epochs)

    query_iterator = iter(query_loader)
    support_iterator = iter(support_loader)
    support_batches = math.ceil(w * s / bs)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    if params.ft_augmentation:
        train_history_path = train_history_path.replace('.csv', '_aug_{}.csv'.format(params.ft_augmentation))
        test_history_path = test_history_path.replace('.csv', '_aug_{}.csv'.format(params.ft_augmentation))
    if params.ft_manifold:
        train_history_path = train_history_path.replace('.csv', '_manifold_{}.csv'.format(params.ft_manifold))
        test_history_path = test_history_path.replace('.csv', '_manifold_{}.csv'.format(params.ft_manifold))
    if params.ft_mixup:
        train_history_path = train_history_path.replace('.csv', '_mixup_{}.csv'.format(params.ft_mixup))
        test_history_path = test_history_path.replace('.csv', '_mixup_{}.csv'.format(params.ft_mixup))
    if params.ft_cutmix:
        train_history_path = train_history_path.replace('.csv', '_cutmix_{}.csv'.format(params.ft_cutmix))
        test_history_path = test_history_path.replace('.csv', '_cutmix_{}.csv'.format(params.ft_cutmix))
    if params.ft_label_smoothing != 0:
        train_history_path = train_history_path.replace('.csv', '_ls.csv')
        test_history_path = test_history_path.replace('.csv', '_ls.csv')

    loss_history_path = train_history_path.replace('train_history', 'loss_history')
    params_path = get_ft_params_path(output_dir)

    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    #print('Saving finetune validation history to {}'.format(train_history_path))
    print()
    # saving parameters on this json file
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)
    
    # 저장할 dataframe
    df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    # df_grad = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
    #                        columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])

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

    print('Starting fine-tune')
    if use_fixed_features:
        print('Running optimized fixed-feature fine-tuning (no augmentation, fixed body)')
    print()

    for episode in range(n_episodes):
        # Reset models for each episode
        if not torch_pretrained:
            body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
        head = get_classifier_head_class(params.ft_head)(512, params.n_way,
                                                         params)  # TODO: apply ft_features
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
        criterion = nn.CrossEntropyLoss(label_smoothing=params.ft_label_smoothing).cuda()
        ewc = ElasticWeightConsolidation(head, criterion, optimizer)

        # Labels are always [0, 0, ..., 1, ..., w-1]
        # x : input img, f : extracted feature of x, y : label
        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]

        x_query = next(query_iterator)[0].cuda()
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).cuda() 

        if use_fixed_features:  # load data and extract features once per episode
            with torch.no_grad():
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

                # body를 통과한 extracted feature
                # f_support shape : (25, 512)
                if torch_pretrained:
                    f_support = backbone(x_support)
                    f_query = backbone(x_query)
                else:
                    f_support = body.forward_features(x_support, params.ft_features)
                    f_query = body.forward_features(x_query, params.ft_features)
                    

        train_acc_history = []
        test_acc_history = []
        train_loss_history = []
        train_grad_history = []

        n_iter = 0
        for epoch in range(n_epoch):
            # Train
            body.train()
            head.train()

            if not use_fixed_features:  # load data every epoch
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda() # [25, 3, 224, 224]

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s) # shuffle 5 * 5 or 1 * 5

            v2_bool = epoch < 30 and (params.ft_mixup == "v2" or params.ft_cutmix == "v2")
            v2_reverse_bool = epoch >= 70 and (params.ft_mixup == "v2_reverse" or params.ft_cutmix == "v2_reverse")
            mix_bool = (params.ft_mixup or params.ft_cutmix) and not v2_bool and not v2_reverse_bool
            x_support_copy = copy.deepcopy(x_support)

            # cutmix & mixup
            if mix_bool:
                lam = np.random.beta(1.0, 1.0) # 어차피 Uniform sampling
                bbx1, bby1, bbx2, bby2 = rand_bbox(x_support.shape, lam)
                
                rand_index = torch.randperm(x_support.shape[0])
                shuffled_y_support = y_support[rand_index] # shuffled label
                
            if params.ft_cutmix and mix_bool: # recalculate ratio of img b by its area
                x_support_copy[:,:,bbx1:bbx2, bby1:bby2] = x_support[rand_index,:,bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_support.shape[-1] * x_support.shape[-2]))
            elif params.ft_mixup and mix_bool:
                x_support_copy = lam * x_support[:,:,:] + (1. - lam) * x_support[rand_index,:,:]

            with torch.no_grad():
                f_support_mixed = body.forward_features(x_support_copy, params.ft_features)

            # iteration 25/bs(5shot) or 5/bs(1shot)
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]
                y = y_support[batch_indices] # label of support set

                if use_fixed_features:
                    if mix_bool:
                        f = f_support_mixed[batch_indices]
                        y_mix = shuffled_y_support[batch_indices]
                    else:
                        f = f_support[batch_indices]
                else: # if use augmentation or update body
                    f = body.forward_features(x_support[batch_indices], params.ft_features)

                if (params.ft_mixup == "v3" or params.ft_cutmix == "v3"):
                    f = torch.concat([f_support[batch_indices], f])
                    y = torch.concat([y_support[batch_indices], y_support[batch_indices]])
                    y_mix = torch.concat([y_support[batch_indices], y_mix])

                if torch_pretrained:
                    f = f.squeeze(2).squeeze(2)
                
                # manifold(representation) augmentation
                if params.ft_manifold == 'v1':  # only 2 mins to 0
                    feat_mask = torch.ones_like(f)
                    mins = torch.topk(torch.abs(f), k=2, dim=1, largest=False).indices
                    for i in range(len(mins)):
                        feat_mask[i][mins[i]] = 0
                    f = f * feat_mask
                elif params.ft_manifold == 'v2': # mins for the num of epoch to 0 
                    feat_mask = torch.ones_like(f)
                    mins = torch.topk(torch.abs(f), k=epoch, dim=1, largest=False).indices
                    for i in range(len(mins)):
                        feat_mask[i][mins[i]] = 0
                    f = f * feat_mask
                elif params.ft_manifold == 'v3': # mins for the num of epoch//2 to 0
                    feat_mask = torch.ones_like(f)
                    mins = torch.topk(torch.abs(f), k=epoch//2, dim=1, largest=False).indices
                    for i in range(len(mins)):
                        feat_mask[i][mins[i]] = 0
                    f = f * feat_mask
                elif params.ft_manifold == 'mixup': # mins for the num of epoch//2 to 0
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(x_support.shape[0])
                    y_mix = y[rand_index]
                    f = lam * f[:,:] + (1. - lam) * f[rand_index,:]
                else:
                    if params.ft_manifold is not None:
                        raise ValueError('Invalid version number of manifold augmentation')

                # head 거치기
                p = head(f)

                correct += torch.eq(y, p.argmax(dim=1)).sum()
                if mix_bool or params.ft_manifold == 'mixup':
                    loss = criterion(p, y) * lam + criterion(p, y_mix) * (1. - lam)
                else:
                    loss = criterion(p, y)
                
                if params.ft_EWC:
                    loss = loss + ewc._compute_consolidation_loss(1000000)

                optimizer.zero_grad() # pytorch에서는 이걸 안해주면 gradient를 계속 누적함 (각 Iteration이 끝나면 초기화해줌)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if params.ft_lr_scheduler != "None":
                scheduler.step()
            train_loss = total_loss / support_batches
            train_acc = correct / (w * s)

            # Evaluation
            body.eval()
            head.eval()

            # Test Using Query
            if params.ft_intermediate_test or epoch == n_epoch - 1:
                with torch.no_grad():
                    if not use_fixed_features:
                        f_query = body.forward_features(x_query, params.ft_features)
                    if torch_pretrained:
                        f_query = f_query.squeeze(-1).squeeze(-1)
                    p_query = head(f_query) # (75, 512)
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
            else:
                test_acc = torch.tensor(0)
            
            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())
            train_loss_history.append(train_loss)

        # save model every episode
        # torch.save(head.state_dict(), output_dir+'/{}epoch_head.pt'.format(n_epoch))
        # torch.save(body.state_dict(), output_dir+'/{}epoch_body.pt'.format(n_epoch))

        #print("Total iterations for {} epochs : {}".format(n_epoch, n_iter))
        df_train.loc[episode + 1] = train_acc_history
        df_train.to_csv(train_history_path)
        df_test.loc[episode + 1] = test_acc_history
        df_test.to_csv(test_history_path)
        df_loss.loc[episode + 1] = train_loss_history
        df_loss.to_csv(loss_history_path)

        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))

    print('Saved history to:')
    print(train_history_path)
    print(test_history_path)
    df_train.to_csv(train_history_path)
    df_test.to_csv(test_history_path)
    df_loss.to_csv(loss_history_path)


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
