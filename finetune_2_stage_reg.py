import copy
import json
import math
import os
import pickle
from webbrowser import get
import pandas as pd
import torch.nn as nn
#import torchsummary as summary
from torchvision import transforms
import itertools
from backbone import get_backbone_class
import backbone
from datasets.dataloader import get_episodic_dataloader, get_labeled_episodic_dataloader
from datasets.transforms import rand_bbox
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, get_ft_valid_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path, get_ft_v_score_history_path, \
    get_ft_loss_history_path
from utils import *
import time 
from sklearn.cluster import KMeans 
from sklearn.metrics.cluster import v_measure_score

def main(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    base_output_dir = get_output_directory(params) 
    output_dir = get_ft_output_directory(params)
    torch_pretrained = ("torch" in params.backbone)

    print('Running fine-tune with output folder:')
    print(output_dir)
    
    # Settings
    n_episodes = 600
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot

    n_epoch = 100
    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot

    # Model
    backbone = get_backbone_class(params.backbone)()
    backbone_clean = get_backbone_class(params.backbone)()

    body = get_model_class(params.model)(backbone, params) # f
    body_clean = get_model_class(params.model)(backbone_clean, params) # f*

    if params.ft_features is None:
        pass
    else:
        if params.ft_features not in body.supported_feature_selectors:
            raise ValueError(
                'Feature selector "{}" is not supported for model "{}"'.format(params.ft_features, params.model))

    # Dataloaders
    # Note that both dataloaders sample identical episodes, via episode_seed
    support_epochs = n_epoch
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=params.ft_augmentation,
                                                     unlabeled_ratio=0,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed,
                                                     episode_seed=params.ft_episode_seed)

    support_clean_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=None,
                                                     unlabeled_ratio=0,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed,
                                                     episode_seed=params.ft_episode_seed)

    query_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=False,
                                                   n_query_shot=q, n_episodes=n_episodes, n_epochs=1,
                                                   augmentation=None,
                                                   unlabeled_ratio=0,
                                                   num_workers=params.num_workers,
                                                   split_seed=params.split_seed,
                                                   episode_seed=params.ft_episode_seed,
                                                   eval_mode=params.ft_tta_mode)

    if params.ft_valid_mode:
        valid_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                        n_query_shot=q, n_episodes=n_episodes, n_epochs=1,
                                                        augmentation=None,
                                                        unlabeled_ratio=0,
                                                        num_workers=params.num_workers,
                                                        split_seed=params.split_seed,
                                                        episode_seed=params.ft_episode_seed,
                                                        eval_mode=params.ft_valid_mode)
    else:
        valid_loader = None

    assert (len(support_loader) == n_episodes * support_epochs)
    assert (len(query_loader) == n_episodes)

    support_iterator = iter(support_loader)
    support_clean_iterator = iter(support_clean_loader)
    support_batches = math.ceil(n_data / bs)
    if valid_loader is not None:
        valid_iterator = iter(valid_loader)
    else:
        valid_iterator = None
    query_iterator = iter(query_loader)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir).replace('.csv', '_stage2_{}.csv'.format(params.two_stage_reg_rate))
    loss_history_path = get_ft_loss_history_path(output_dir).replace('.csv', '_stage2_{}.csv'.format(params.two_stage_reg_rate))
    valid_history_path = get_ft_valid_history_path(output_dir).replace('.csv', '_stage2_{}.csv'.format(params.two_stage_reg_rate))
    test_history_path = get_ft_test_history_path(output_dir).replace('.csv', '_stage2_{}.csv'.format(params.two_stage_reg_rate))
    support_v_score_history_path, query_v_score_history_path = get_ft_v_score_history_path(output_dir)

    params_path = get_ft_params_path(output_dir)

    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    if params.ft_valid_mode:
        valid_history_path = valid_history_path.replace(".csv", "_"+params.ft_valid_mode+".csv")
        print('Saving finetune validation history to {}'.format(valid_history_path))
    if params.ft_tta_mode:
        test_history_path = test_history_path.replace(".csv", "_"+params.ft_tta_mode+".csv")
    print('Saving finetune test history to {}'.format(test_history_path))
    print()

    # saving parameters on this json file
    with open(params_path, 'w') as f_batch:
        json.dump(vars(params), f_batch, indent=4)
    l1_dist = nn.PairwiseDistance(p=1) # l1 distance for aux_loss
    # 저장할 dataframe
    df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_valid = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_v_score_support = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                                     columns=['epoch{}'.format(e+1) for e in range(n_epoch)])
    df_v_score_query = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e) for e in range(n_epoch+1)])
    
    # Pre-train state
    if not torch_pretrained:
        if params.ft_pretrain_epoch is None: # best state
            body_state_path = get_final_pretrain_state_path(base_output_dir)
        
        if params.source_dataset == 'tieredImageNet':
            body_state_path = './logs/baseline/output/pretrained_model/tiered/resnet18_base_LS_base/pretrain_state_0090.pt'

        if not os.path.exists(body_state_path):
            raise ValueError('Invalid pre-train state path: ' + body_state_path)

        print('Using pre-train state:', body_state_path)
        print()
        state = torch.load(body_state_path)
        state_clean = torch.load(body_state_path)
    else:
        pass

    # print time
    now = time.localtime()
    start = time.time()
    print("%02d/%02d %02d:%02d:%02d" %(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print()

    # mixup, cutmix, manifold mixup need 2 labels <- mix_bool == True
    mix_bool = (params.ft_mixup or params.ft_cutmix or params.ft_manifold_mixup)
    # for cutmix or mixup (different class option)
    if mix_bool:
        all_cases = list(itertools.permutations(list(range(w))))
        class_shuffled = all_cases
        for case in copy.deepcopy(all_cases):
            for idx in range(w):
                if case[idx] == idx:
                    class_shuffled.remove(case)
                    break

    # For each episode
    for episode in range(n_episodes):
        # Reset models for each episode
        if not torch_pretrained:
            body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
            body_clean.load_state_dict(copy.deepcopy(state_clean))
        else:
            body = get_model_class(params.model)(backbone, params)
            body_clean = get_model_class(params.model)(backbone, params)

        head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features
        head_clean = get_classifier_head_class(params.ft_head)(512, params.n_way, params)

        body.cuda()
        head.cuda()

        body_clean.cuda()
        head_clean.cuda()

        opt_params = []
        opt_params.append({'params': head.parameters()})
        opt_params.append({'params': body.parameters()})
        optimizer = torch.optim.SGD(opt_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)

        opt_clean_params = []
        opt_clean_params.append({'params': head_clean.parameters()})
        opt_clean_params.append({'params': body_clean.parameters()})
        optimizer_clean = torch.optim.SGD(opt_clean_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)

        criterion = nn.CrossEntropyLoss().cuda()
        criterion_clean = nn.CrossEntropyLoss().cuda()

        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]
        y_clean_support = torch.arange(w).repeat_interleave(s).cuda()
        y_support_np = y_support.cpu().numpy()

        if valid_iterator is not None:
            f_valid = None
            y_valid = torch.arange(w).repeat_interleave(s).cuda()
            y_valid_np = y_valid.cpu().numpy()
            if 'fixed' in params.ft_valid_mode :
                x_valid = torch.cat(next(valid_iterator)[0]).cuda()
                y_valid = torch.cat([y_valid]*(len(x_valid)//len(y_valid)), dim=0)
            else:
                x_valid = next(valid_iterator)[0].cuda()

        if params.ft_tta_mode and 'fixed' in params.ft_tta_mode :
                x_query = torch.cat(next(query_iterator)[0]).cuda()
        else:
            x_query = next(query_iterator)[0].cuda()

        y_query = torch.arange(w).repeat_interleave(q).cuda() 
        f_query = None
        y_query_np = y_query.cpu().numpy()
        num_aug = len(x_query)//len(y_query)
        
        train_acc_history = []
        train_loss_history = []
        valid_acc_history = []
        test_acc_history = []
        support_v_score = []
        query_v_score = []

        # V-measure support and query for epoch 0
        if s != 1:
            support_v_score.append(0.0)

        with torch.no_grad():
            f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
            if params.ft_tta_mode and 'fixed' in params.ft_tta_mode :
                f_query = torch.mean(torch.cat(torch.chunk(f_query.unsqueeze(0), num_aug, dim=1), axis=0), axis=0)
            f_query_np = f_query.cpu().numpy()
            kmeans = KMeans(n_clusters = w)
            cluster_pred = kmeans.fit(f_query_np).labels_
            query_v_score.append(v_measure_score(cluster_pred, y_query_np))

        # Stage One : train f with s
        for epoch in range(n_epoch):
            # Train
            body_clean.train()
            head_clean.train()

            x_clean_support = next(support_clean_iterator)[0].cuda()

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s) 

            # For each iteration
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]

                y_clean_batch = y_clean_support[batch_indices] 
                f_clean_batch = body_forward(x_clean_support[batch_indices], body_clean, backbone_clean, torch_pretrained, params)
                pred_clean = head_clean(f_clean_batch)

                correct += torch.eq(y_clean_batch, pred_clean.argmax(dim=1)).sum()

                #aux_loss = l1_dist(f_clean_batch, f_batch).mean() # l1 distance
                
                loss_clean = criterion_clean(pred_clean, y_clean_batch)

                optimizer_clean.zero_grad() 
                loss_clean.backward() 
                optimizer_clean.step()

                total_loss += loss_clean.item()

            train_loss = total_loss / support_batches
            train_acc = correct / n_data

            if epoch == n_epoch - 1:
                body.eval()
                head.eval()          
                with torch.no_grad():      
                    # f_support = body_forward(x_clean_support, body, backbone, torch_pretrained, params)                  
                    f_query = body_forward(x_query, body_clean, backbone_clean, torch_pretrained, params)
                    pred = head_clean(f_query)
                    if params.ft_tta_mode and 'fixed' in params.ft_tta_mode :
                        pred = torch.mean(torch.cat(torch.chunk(pred.unsqueeze(0), num_aug, dim=1), axis=0), axis=0)
                    correct = torch.eq(y_query, pred.argmax(dim=1)).sum()
                test_acc = correct / pred.shape[0]

        fmt = 'Stage 1 Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc * 100, test_acc * 100))

        # Second Stage
        for epoch in range(n_epoch):
            # Train
            body.train()
            head.train()

            body_clean.eval()
            head_clean.eval()

            aug_bool = mix_bool or params.ft_augmentation
            if params.ft_scheduler_end is not None: # if aug is scheduled, 
                aug_bool = (epoch < params.ft_scheduler_end and epoch >= params.ft_scheduler_start) and aug_bool

            x_support = next(support_iterator)[0].cuda()

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s) 

            if aug_bool:
                x_support_aug = copy.deepcopy(x_support)
                if mix_bool:
                    if params.ft_mixup:
                        mode = params.ft_mixup
                    elif params.ft_cutmix:
                        mode = params.ft_cutmix
                    elif params.ft_manifold_mixup:
                        mode = params.ft_manifold_mixup
                    else:
                        raise ValueError ("Unknown mode: %s" % mode)
                    
                    # lambda options
                    if mode != 'lam':
                        lam = np.random.beta(1.0, 1.0) 
                    else: # mode == 'lam'
                        lam = np.random.beta(0.01*(epoch+1), 0.01*(epoch+1))
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x_support.shape, lam) # cutmix corner points

                    if mode == 'both' or mode == 'lam':
                        indices_shuffled = torch.randperm(x_support.shape[0])
                    else:
                        shuffled = np.array([])
                        if mode == 'same' :
                            class_arr = range(w)
                        elif mode == 'diff': 
                            class_arr_idx = np.random.choice(range(len(class_shuffled)), 1)[0]
                            class_arr = class_shuffled[class_arr_idx]

                        for clss in class_arr:
                            shuffled = np.append(shuffled, np.random.permutation(range(clss*s, (clss+1)*s))) 
                        indices_shuffled = torch.from_numpy(shuffled).long()   

                    # mixup
                    if params.ft_mixup:
                        x_support_aug = lam * x_support[:,:,:] + (1. - lam) * x_support[indices_shuffled,:,:]
                        
                    # cutmix
                    elif params.ft_cutmix: # recalculate ratio of img b by its area
                        x_support_aug[:,:,bbx1:bbx2, bby1:bby2] = x_support[indices_shuffled,:,bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_support.shape[-1] * x_support.shape[-2])) # adjust lambda
                    
                    y_shuffled = y_support[indices_shuffled]

            # For each iteration
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]

                y_batch = y_support[batch_indices]
				
                if aug_bool and mix_bool: # cutmix, mixup
                    y_shuffled_batch = y_shuffled[batch_indices]

                f_batch = body_forward(x_support_aug[batch_indices], body, backbone, torch_pretrained, params)
                if params.ft_manifold_mixup:
                    f_batch_shuffled = body_forward(x_support[indices_shuffled[batch_indices]], body, backbone, torch_pretrained, params)
                    f_batch = lam * f_batch[:,:] + (1. - lam) * f_batch_shuffled[:,:]
                    
                f_clean_batch = body_forward(x_clean_support[batch_indices], body_clean, backbone_clean, torch_pretrained, params)

                # head 거치기
                pred = head(f_batch)

                correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()

                #aux_loss = l1_dist(f_clean_batch, f_batch).mean() # l1 distance
                #loss_clean = criterion_clean(pred_clean, y_clean_batch)

                loss_aug = criterion(pred, y_batch)

                distance = l1_dist(f_clean_batch.detach(), f_batch.detach()).mean()

                if aug_bool and mix_bool:
                    loss_aug = criterion(pred, y_batch) * lam + criterion(pred, y_shuffled_batch) * (1. - lam)
                    
                loss = params.two_stage_reg_rate * loss_aug + (1-params.two_stage_reg_rate) * distance

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / support_batches
            train_acc = correct / n_data

            if params.ft_intermediate_test or epoch == n_epoch - 1:
                body.eval()
                head.eval()
                # Validation Set Evaluation
                if params.ft_valid_mode:
                    with torch.no_grad():
                        f_valid = body_forward(x_valid, body, backbone, torch_pretrained, params)
                        pred = head(f_valid)
                        correct = torch.eq(y_valid, pred.argmax(dim=1)).sum()
                    valid_acc = correct / pred.shape[0]
                else:
                    valid_acc = torch.tensor(0)

                # V-measure support
                # if params.v_score and params.n_shot != 1:
                #     with torch.no_grad():
                #         f_support = body_forward(x_support, body, backbone, torch_pretrained, params)
                #     f_support = f_support.cpu().numpy()
                #     kmeans = KMeans(n_clusters = w)
                #     cluster_pred = kmeans.fit(f_support).labels_
                #     support_v_score.append(v_measure_score(cluster_pred, y_support_np))

                # Test (Query) Set Evaluation                
                with torch.no_grad():      
                    # f_support = body_forward(x_clean_support, body, backbone, torch_pretrained, params)                  
                    f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
                    pred = head(f_query)
                    if params.ft_tta_mode and 'fixed' in params.ft_tta_mode :
                        pred = torch.mean(torch.cat(torch.chunk(pred.unsqueeze(0), num_aug, dim=1), axis=0), axis=0)
                    correct = torch.eq(y_query, pred.argmax(dim=1)).sum()
                test_acc = correct / pred.shape[0]
                # Query V-measure
                if params.v_score:
                    f_query = f_query.cpu().numpy()
                    kmeans = KMeans(n_clusters = w)
                    cluster_pred = kmeans.fit(f_query).labels_
                    query_v_score.append(v_measure_score(cluster_pred, y_query_np))
            else:
                valid_acc = torch.tensor(0)
                test_acc = torch.tensor(0)
                support_v_score.append(0.0)
                query_v_score.append(0.0)
            
            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_acc_history.append(train_acc.item())
            valid_acc_history.append(valid_acc.item())
            test_acc_history.append(test_acc.item())
            train_loss_history.append(train_loss)

        df_train.loc[episode + 1] = train_acc_history
        df_train.to_csv(train_history_path)
        df_test.loc[episode + 1] = test_acc_history
        df_test.to_csv(test_history_path)
        df_loss.loc[episode + 1] = train_loss_history
        df_loss.to_csv(loss_history_path)

        if params.ft_valid_mode:
            df_valid.loc[episode + 1] = valid_acc_history
            df_valid.to_csv(valid_history_path)

        if params.v_score:
            if params.n_shot != 1:
                df_v_score_support.loc[episode + 1] = support_v_score
                df_v_score_support.to_csv(support_v_score_history_path)
            df_v_score_query.loc[episode + 1] = query_v_score
            df_v_score_query.to_csv(query_v_score_history_path)

        if params.ft_valid_mode:
            df_valid.loc[episode + 1] = valid_acc_history
            fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} valid_acc={:6.2f} test_acc={:6.2f}'
            print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, valid_acc_history[-1] * 100, test_acc_history[-1] * 100))
        else: 
            fmt = 'Stage 2 Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
            print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))
    end = time.time()

    print('Saved history to:')
    print(train_history_path)
    print(test_history_path)
    df_train.to_csv(train_history_path)
    df_test.to_csv(test_history_path)
    df_loss.to_csv(loss_history_path)
    if params.ft_valid_mode:
        df_valid.to_csv(valid_history_path)
    print("\nIt took {:6.2f} min to finish current training\n".format((end-start)/60))

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
