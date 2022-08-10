import copy
import json
import math
import os
import pickle
from webbrowser import get
import pandas as pd
import torch.nn as nn
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
    #os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    device = torch.device(f'cuda:{params.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    base_output_dir = get_output_directory(params) 
    output_dir = get_ft_output_directory(params)
    torch_pretrained = ("torch" in params.backbone)

    assert (params.ft_augmentation is not None)

    print('Running fine-tune with output folder:')
    print(output_dir)
    
    # Settings
    n_episodes = 600
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot

    n_epoch = params.ft_epochs
    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot

    if params.ft_features is None:
        pass
    else:
        if params.ft_features not in body.supported_feature_selectors:
            raise ValueError(
                'Feature selector "{}" is not supported for model "{}"'.format(params.ft_features, params.model))
    
    if params.ft_augmentation == "randomresizedcrop":
        aug_type = "rcrop_"
        aug_intensity = [0.2, 0.4, 0.6, 0.8]
    
    elif params.ft_augmentation == "base":
        aug_type = "base_"
        aug_intensity = [1, 2, 3, 4]
        
    elif params.ft_augmentation == "randomcolorjitter":
        aug_type = "cjitter_"
        aug_intensity = [0.2, 0.6, 0.8, 1.0]
        
    num_intensity = len(aug_intensity)
    
    for intensity in aug_intensity:
        aug_param = aug_type + str(intensity)

        # Output (history, params)
        train_history_path = get_ft_train_history_path(output_dir).replace('.csv', '_{}.csv'.format(intensity))
        loss_history_path = get_ft_loss_history_path(output_dir).replace('.csv', '_{}.csv'.format(intensity))
        test_history_path = get_ft_test_history_path(output_dir).replace('.csv', '_{}.csv'.format(intensity))
        params_path = get_ft_params_path(output_dir)

        print('Saving finetune params to {}'.format(params_path))
        print('Saving finetune train history to {}'.format(train_history_path))
        print('Saving finetune test history to {}'.format(test_history_path))
        print()

        # saving parameters on this json file
        with open(params_path, 'w') as f_batch:
            json.dump(vars(params), f_batch, indent=4)
        
        # 저장할 dataframe
        df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                                columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
        df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
        df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])

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
        else:
            pass

        # Model
        backbone = get_backbone_class(params.backbone)()
        body = get_model_class(params.model)(backbone, params)

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

            # Dataloaders
        # Note that both dataloaders sample identical episodes, via episode_seed
        support_epochs = n_epoch
        support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                        n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                        augmentation=aug_param,
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
                                                    episode_seed=params.ft_episode_seed)

        assert (len(support_loader) == n_episodes * support_epochs)
        assert (len(query_loader) == n_episodes)

        support_iterator = iter(support_loader)
        support_batches = math.ceil(n_data / bs)
        query_iterator = iter(query_loader)

        # For each episode
        for episode in range(n_episodes):
            # Reset models for each episode
            if not torch_pretrained:
                body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
            else:
                body = get_model_class(params.model)(copy.deepcopy(backbone), params)
                            
            head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features

            body.cuda()
            head.cuda()

            opt_params = []
            opt_params.append({'params': head.parameters()})
            opt_params.append({'params': body.parameters()})
            optimizer = torch.optim.SGD(opt_params, lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

            criterion = nn.CrossEntropyLoss().cuda()

            x_support = None
            f_support = None
            y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]
            y_support_np = y_support.cpu().numpy()
            x_query = next(query_iterator)[0].cuda()

            y_query = torch.arange(w).repeat_interleave(q).cuda() 
            f_query = None
            y_query_np = y_query.cpu().numpy()
            num_aug = len(x_query)//len(y_query)
            
            train_acc_history = []
            train_loss_history = []
            test_acc_history = []

            with torch.no_grad():
                f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
                if params.ft_tta_mode and 'fixed' in params.ft_tta_mode :
                    f_query = torch.mean(torch.cat(torch.chunk(f_query.unsqueeze(0), num_aug, dim=1), axis=0), axis=0)
                f_query_np = f_query.cpu().numpy()
                kmeans = KMeans(n_clusters = w)
                cluster_pred = kmeans.fit(f_query_np).labels_

            # For each epoch
            for epoch in range(n_epoch):
                # Train
                body.train()
                head.train()

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

                    y_batch = y_support[batch_indices] # label

                    if aug_bool and mix_bool: # cutmix, mixup
                        y_shuffled_batch = y_shuffled[batch_indices]


                    if aug_bool:
                        f_batch = body_forward(x_support_aug[batch_indices], body, backbone, torch_pretrained, params)
                        if params.ft_manifold_mixup:
                            f_batch_shuffled = body_forward(x_support[indices_shuffled[batch_indices]], body, backbone, torch_pretrained, params)
                            f_batch = lam * f_batch[:,:] + (1. - lam) * f_batch_shuffled[:,:]
                    else:
                        f_batch = body_forward(x_support[batch_indices], body, backbone, torch_pretrained, params)

                    # head 거치기
                    pred = head(f_batch)

                    correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()

                    if aug_bool and mix_bool:
                        loss = criterion(pred, y_batch) * lam + criterion(pred, y_shuffled_batch) * (1. - lam)
                    else:
                        loss = criterion(pred, y_batch)

                    optimizer.zero_grad() 
                    loss.backward() 
                    optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / support_batches
                train_acc = correct / n_data

                if params.ft_intermediate_test or epoch == n_epoch - 1:
                    body.eval()
                    head.eval()

                    # Test (Query) Set Evaluation                
                    with torch.no_grad():      
                        # f_support = body_forward(x_clean_support, body, backbone, torch_pretrained, params)                  
                        f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
                        pred = head(f_query)
                        correct = torch.eq(y_query, pred.argmax(dim=1)).sum()
                    test_acc = correct / pred.shape[0]
                else:
                    test_acc = torch.tensor(0)
                
                print_epoch_logs = False
                if print_epoch_logs and (epoch + 1) % 10 == 0:
                    fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                    print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

                train_acc_history.append(train_acc.item())
                test_acc_history.append(test_acc.item())
                train_loss_history.append(train_loss)

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
