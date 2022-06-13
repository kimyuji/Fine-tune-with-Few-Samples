import copy
import json
import math
import os
import pickle
import pandas as pd
import torch.nn as nn
import torchsummary as summary
from torchvision import transforms
import itertools
from backbone import get_backbone_class
import backbone
from datasets.dataloader import get_labeled_episodic_dataloader
from datasets.transforms import transforms_ss
from io_utils import parse_args
from model import get_model_class
from model.simclr import NTXentLoss
from model.supcon import SupConLoss
from model.supcon import FT_SupConLoss
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path, get_ft_v_score_history_path, \
    get_ft_loss_history_path, get_ft_clean_history_path
from utils import *
import time 

#from elastic_weight_consolidation import ElasticWeightConsolidation

from sklearn.cluster import KMeans 
from sklearn.metrics.cluster import v_measure_score

def main(params):
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

    n_epoch = 100
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
    loss_history_path = get_ft_loss_history_path(output_dir)
    support_v_score_history_path, query_v_score_history_path = get_ft_v_score_history_path(output_dir)

    if params.ft_SS:
        train_history_path = train_history_path.replace(".csv", "_{}.csv".format(params.ft_tau))
        test_history_path = test_history_path.replace(".csv", "_{}.csv".format(params.ft_tau))

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
    df_v_score_support = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e+1) for e in range(n_epoch)])
    df_v_score_query = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e) for e in range(n_epoch+1)])

    # Pre-train state
    
    if not torch_pretrained : 
        if params.ft_pretrain_epoch is None: # best state
            body_state_path = get_final_pretrain_state_path(base_output_dir)
        else: # 원하는 epoch수의 state를 받아오고 싶다면 
            body_state_path = get_pretrain_state_path(base_output_dir, params.ft_pretrain_epoch)
        
        if params.source_dataset == 'tieredImageNet':
            body_state_path = './logs/baseline/output/pretrained_model/tiered/resnet18_base_LS_base/pretrain_state_0090.pt'

        if not os.path.exists(body_state_path):
            raise ValueError('Invalid pre-train state path: ' + body_state_path)

        print('Using pre-train state:')
        print(body_state_path)
        print()
        state = torch.load(body_state_path)

    # print time
    now = time.localtime()
    start = time.time()
    print("%02d/%02d %02d:%02d:%02d" %(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print()


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

        # select optimizer
        optimizer = torch.optim.SGD(opt_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        # Loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=params.ft_label_smoothing).cuda()
        simclr_criterion = NTXentLoss(temperature=params.ft_tau, use_cosine_similarity=True)
        supcon_criterion = SupConLoss(temperature=params.ft_tau, base_temperature=params.ft_tau)
        ft_supcon_criterion = FT_SupConLoss(temperature=params.ft_tau, base_temperature=params.ft_tau)

        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]
        cluster_y_support = y_support.cpu().numpy()

        x_query = next(query_iterator)[0].cuda()
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).cuda() 
        cluster_y_query = y_query.cpu().numpy()
        
        train_acc_history = []
        test_acc_history = []
        train_loss_history = []
        support_v_score = []
        query_v_score = []

        if use_fixed_features:  # load data and extract features once per episode
            with torch.no_grad():
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

                if torch_pretrained:
                    f_support = backbone(x_support)
                    f_query = backbone(x_query)
                else:
                    f_support = body.forward_features(x_support, params.ft_features)
                    f_query = body.forward_features(x_query, params.ft_features)
        else:
            # V-measure query for epoch0, augmentation and full
            with torch.no_grad():
                f_query = body.forward_features(x_query, params.ft_features)
                f_query_0 = f_query.cpu().numpy()
                kmeans = KMeans(n_clusters = w)
                cluster_pred = kmeans.fit(f_query_0).labels_
                query_v_score.append(v_measure_score(cluster_pred, cluster_y_query))


#############################################################################################################
        for epoch in range(n_epoch):
            # Train
            body.train()
            head.train()

            if params.ft_update_scheduler == "LP-FT":
                if epoch == 0: 
                    optimizer.param_groups[1]['lr'] = 0.0 # body
                elif epoch == 50: 
                    optimizer.param_groups[1]['lr'] = 0.01
            elif params.ft_update_scheduler == "body-FT":
                if epoch == 0: 
                    optimizer.param_groups[0]['lr'] = 0.0 # head
                elif epoch == 50: 
                    optimizer.param_groups[0]['lr'] = 0.01
            elif params.ft_update_scheduler == "body-LP":
                if epoch == 0: 
                    optimizer.param_groups[0]['lr'] = 0.0 # head
                elif epoch == 50: 
                    optimizer.param_groups[0]['lr'] = 0.01
                    optimizer.param_groups[1]['lr'] = 0.0 # body 
            elif params.ft_update_scheduler == "FT-LP":
                if epoch == 50: 
                    optimizer.param_groups[1]['lr'] = 0.0 # body 

            if not use_fixed_features:  
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s) 

################################################################################################################
            # iteration 25/bs(5shot) or 5/bs(1shot)
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]

                y_batch = y_support[batch_indices] # label

                if use_fixed_features:
                    f_batch = f_support[batch_indices]
                else: # full update
                    f_batch = body.forward_features(x_support[batch_indices], params.ft_features)
                    if params.ft_SS:
                        x_support_ss_1 = transforms_ss(x_support)
                        x_support_ss_2 = transforms_ss(x_support)
                        z1 = body.forward_features(x_support_ss_1[batch_indices], params.ft_features)
                        z2 = body.forward_features(x_support_ss_2[batch_indices], params.ft_features)
                        if params.ft_SS == 'add_supcon':
                            features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
            
                # if torch_pretrained:
                #     f_batch = f_batch.squeeze(2).squeeze(2)
                
                # head 거치기
                pred = head(f_batch)

                correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()
                loss = criterion(pred, y_batch)
                if params.ft_SS == "add_simclr":
                    loss_ss = simclr_criterion(z1, z2)
                    loss = loss + loss_ss
                elif params.ft_SS == "add_supcon":
                    loss_ss = supcon_criterion(features, y_batch)
                    loss = loss + loss_ss
                elif params.ft_SS == "add_ft_supcon":
                    loss_ss = supcon_criterion(features, y_batch)
                    loss = loss + loss_ss

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

                total_loss += loss.item()

                # if params.ft_lr_scheduler:
                #     scheduler.step()

################################################################################################################
            train_loss = total_loss / support_batches
            train_acc = correct / n_data

            # Evaluation
            body.eval()
            head.eval()
            
            # V-measure support
            if params.v_score and params.n_shot != 1:
                with torch.no_grad():
                    f_support = body.forward_features(x_support, params.ft_features)
                f_support = f_support.cpu().numpy()
                kmeans = KMeans(n_clusters = w)
                cluster_pred = kmeans.fit(f_support).labels_
                support_v_score.append(v_measure_score(cluster_pred, cluster_y_support))


            # Test Using Query
            if params.ft_intermediate_test or epoch == n_epoch - 1:
                with torch.no_grad():
                    if not use_fixed_features:
                        if not torch_pretrained:
                            f_query = body.forward_features(x_query, params.ft_features)
                        else:
                            f_query = backbone(x_query)
                            f_query = f_query.squeeze(-1).squeeze(-1)
                    p_query = head(f_query) 
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
                # V-measure query
                if params.v_score:
                    f_query = f_query.cpu().numpy()
                    kmeans = KMeans(n_clusters = w)
                    cluster_pred = kmeans.fit(f_query).labels_
                    query_v_score.append(v_measure_score(cluster_pred, cluster_y_query))
            else:
                test_acc = torch.tensor(0)
            
            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())
            train_loss_history.append(train_loss)
################################################################################################################
        
        df_train.loc[episode + 1] = train_acc_history
        df_train.to_csv(train_history_path)
        df_test.loc[episode + 1] = test_acc_history
        df_test.to_csv(test_history_path)
        df_loss.loc[episode + 1] = train_loss_history
        df_loss.to_csv(loss_history_path)

        if params.v_score:
            if params.n_shot != 1:
                df_v_score_support.loc[episode + 1] = support_v_score
                df_v_score_support.to_csv(support_v_score_history_path)
            df_v_score_query.loc[episode + 1] = query_v_score
            df_v_score_query.to_csv(query_v_score_history_path)

        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
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
    print("\nIt took {:6.2f} to finish current training\n".format((end-start)/60))


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
