import copy
import json
import math
import os
import pickle
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
    n_episodes = 100
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
    backbone_pre = get_backbone_class(params.backbone)() 
    
    body = get_model_class(params.model)(backbone, params)
    pretrained = get_model_class(params.model)(backbone_pre, params)

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

    layer_path_l2 = train_history_path.replace('train_history', 'layer_diff_l2')
    layer_path_sub = train_history_path.replace('train_history', 'layer_diff_sub')

    params_path = get_ft_params_path(output_dir)

    print('\nSaving layer difference output to {}'.format(layer_path_l2))
    #print('Saving finetune validation history to {}'.format(train_history_path))
    print()
    # saving parameters on this json file
    with open(params_path, 'w') as f_batch:
        json.dump(vars(params), f_batch, indent=4)
    
    # 저장할 dataframe
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])

    if not torch_pretrained : 
        if params.ft_pretrain_epoch is None: # best state
            body_state_path = get_final_pretrain_state_path(base_output_dir)
        else: # 원하는 epoch수의 state를 받아오고 싶다면 
            body_state_path = get_pretrain_state_path(base_output_dir, params.ft_pretrain_epoch)
        
        if params.source_dataset == 'tieredImageNet':
            body_state_path = './logs/baseline/output/pretrained_model/tiered/resnet18_base_LS_base/pretrain_state_0090.pt'

        if not os.path.exists(body_state_path):
            raise ValueError('Invalid pre-train state path: ' + body_state_path)

        print()
        state = torch.load(body_state_path)


    all_cases = list(itertools.permutations(list(range(w))))
    class_shuffled = all_cases
    for case in copy.deepcopy(all_cases):
        for idx in range(w):
            if case[idx] == idx : 
                class_shuffled.remove(case)
                break 

    pretrained.load_state_dict(state)
    pretrained.eval()

    layer_name = []
    for p in  pretrained.named_parameters():
        if 'fc' not in p[0]:
            layer_name.append(p[0])
    # 저장할 dataframe
    df_layer_l2 = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=layer_name)
    df_layer_sub = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=layer_name)
########################################################################################################################
    for episode in range(n_episodes):
        # Reset models for each episode
        if params.ft_no_pretrain:
            backbone = get_backbone_class(params.backbone)() 
            body = get_model_class(params.model)(backbone, params)
            pretrained = copy.deepcopy(body)
            pretrained.eval()
        else:
            if not torch_pretrained:
                body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.

        head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features
        rand_init = copy.deepcopy(head)
        rand_init.eval()

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

        if use_fixed_features:  # load data and extract features once per episode
            with torch.no_grad():
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

                f_support = body_forward(x_support, body, backbone, torch_pretrained, params)
                f_query = body_forward(x_query, body, backbone, torch_pretrained, params)

                if params.ft_clean_test: # no aug, head
                    f_support_clean = copy.deepcopy(f_support)

        train_acc_history = []
        test_acc_history = []
        train_loss_history = []
        train_acc_clean_history = []
        support_v_score = []
        query_v_score = []

#############################################################################################################
        for epoch in range(n_epoch):
            # Train
            body.train()
            head.train()

            # 4 augmentation methods : mixup, cutmix, manifold, augmentation(transform)
            # mixup, cutmix, manifold mixup need 2 labels <- mix_bool == True
            mix_bool = (params.ft_mixup or params.ft_cutmix or params.ft_manifold_mixup) 
            aug_bool = mix_bool or params.ft_augmentation
            if params.ft_scheduler_end is not None: # if aug is scheduled, 
                aug_bool = (epoch < params.ft_scheduler_end and epoch >= params.ft_scheduler_start) and aug_bool

            if not use_fixed_features:  
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda() 
                if params.ft_clean_test:
                    x_support_clean, _ = next(support_iterator_clean)
                    x_support_clean = x_support_clean.cuda()
                    if params.ft_augmentation:
                        f_support_clean = body_forward(x_support_clean, body, backbone, torch_pretrained, params)


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

                    if mode != 'lam':
                        lam = np.random.beta(1.0, 1.0) 
                    else:
                        lam = np.random.beta(0.01*(epoch+1), 0.01*(epoch+1))
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x_support.shape, lam)

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

                    with torch.no_grad():
                        f_support = body_forward(x_support_aug, body, backbone, torch_pretrained, params)

################################################################################################################
            # iteration 25/bs(5shot) or 5/bs(1shot)
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]

                y_batch = y_support[batch_indices] # label
                if aug_bool and mix_bool:
                    y_shuffled_batch = y_shuffled[batch_indices]

                if use_fixed_features:
                    f_batch = f_support[batch_indices]
                else: # if use augmentation or update body
                    if aug_bool:   
                        f_batch = body_forward(x_support_aug[batch_indices], body, backbone, torch_pretrained, params)
                    else:
                        f_batch = body_forward(x_support[batch_indices], body, backbone, torch_pretrained, params)
                    if params.ft_manifold_mixup:
                        f_batch_shuffled = body_forward(x_support[indices_shuffled[batch_indices]], body, backbone, torch_pretrained, params)

                if params.ft_manifold_mixup:
                    f_batch = lam * f_batch[:,:] + (1. - lam) * f_batch_shuffled[:,:]
                
                # head 거치기
                pred = head(f_batch)

                correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()

                if aug_bool and mix_bool:
                    loss = criterion(pred, y_batch) * lam + criterion(pred, y_shuffled_batch) * (1. - lam)
                else:
                    loss = criterion(pred, y_batch)
                
                # if params.ft_EWC:
                #     loss = loss + ewc._compute_consolidation_loss(1000000)

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

            # Test Using Clean Support
            if params.ft_clean_test:
                with torch.no_grad():
                    if params.ft_parts == 'head':
                        pred_clean = head(f_support_clean)
                    elif params.ft_parts != 'head':
                        f_support_clean = body_forward(x_support_clean, body, backbone, torch_pretrained, params)
                        pred_clean = head(f_support_clean)
                train_acc_clean = torch.eq(y_support, pred_clean.argmax(dim=1)).sum() / n_data
                train_acc_clean_history.append(train_acc_clean.item())
            
            # Calculate V measurement score
            if params.v_score and params.n_shot != 1:
                with torch.no_grad():
                    f_support = body_forward(x_support, body, backbone, torch_pretrained, params)
                f_support = f_support.cpu().numpy()
                cluster_label = y_support.cpu().numpy()
                kmeans = KMeans(n_clusters = w)
                cluster_pred = kmeans.fit(f_support).labels_
                support_v_score.append(v_measure_score(cluster_pred, cluster_label))


            # Test Using Query
            if epoch == n_epoch - 1:
                with torch.no_grad():
                    if not use_fixed_features:
                        f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
                    p_query = head(f_query) 
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
                if params.v_score:
                    f_query = f_query.cpu().numpy()
                    cluster_label = y_query.cpu().numpy()
                    kmeans = KMeans(n_clusters = w)
                    cluster_pred = kmeans.fit(f_query).labels_
                    query_v_score.append(v_measure_score(cluster_pred, cluster_label))
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
        if params.save_LP_FT_feat: 
            pretrained.cuda()
            with torch.no_grad():
                lp_feat = pretrained.forward_features(x_query, params.ft_features)

            feature_path = os.path.join('./feature', params.target_dataset) # NOTE : no augmentation applicable
            feature_path = os.path.join(feature_path, "{:03d}shot_full".format(s)) 
            #feature_path = os.path.join(feature_path, 'flip')
            os.makedirs(feature_path, exist_ok=True)
            np.save(feature_path+'/lp_{:0.2f}.npy'.format(test_acc*100), lp_feat.cpu().numpy())
            np.save(feature_path+'/ft_{:0.2f}.npy'.format(test_acc*100), f_query.cpu().numpy())

        body.cpu()
        pretrained.cpu()
        head.cpu()
        layer_diff_l2 = []
        layer_diff_sub = []

        # layer diff L2 norm
        for p, b in zip(pretrained.named_parameters(), body.named_parameters()):
            if 'fc' not in p[0] and 'classifier' not in p[0]:
                lp_body = p[1].cpu().detach().numpy()
                ft_body = b[1].cpu().detach().numpy()
                layer_diff_l2.append(np.linalg.norm(lp_body-ft_body))
                layer_diff_sub.append((np.abs(lp_body-ft_body) / np.abs(lp_body)).mean())
        for r, c in zip(rand_init.named_parameters(), head.named_parameters()):
            lp_head = r[1].cpu().detach().numpy()
            ft_head = c[1].cpu().detach().numpy()
            layer_diff_l2.append(np.linalg.norm(lp_head-ft_head))
            layer_diff_sub.append((np.abs(lp_head-ft_head) / np.abs(lp_head)).mean())
        df_layer_l2.loc[episode+1] = layer_diff_l2
        df_layer_sub.loc[episode+1] = layer_diff_sub

        #print("Total iterations for {} epochs : {}".format(n_epoch, n_iter))
        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    print("Training 100 episodes and calculating layer difference finished")
    #print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))


    print('Saving layer difference ...')
    df_layer_l2.to_csv(layer_path_l2)
    df_layer_sub.to_csv(layer_path_sub)


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