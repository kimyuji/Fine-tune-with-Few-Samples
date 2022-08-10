import os
import copy
import json
import math
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    device = torch.device(f'cuda:{params.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    print(f"\nCurrently Using GPU {device}\n")
    base_output_dir = get_output_directory(params) 
    output_dir = get_ft_output_directory(params)
    torch_pretrained = ("torch" in params.backbone)

    upt_dir = '_'.join(params.upt_blocks)
    output_dir = output_dir.replace('full', 'ftlp')
    os.makedirs(output_dir, exist_ok=True)
    
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

    # Model
    backbone = get_backbone_class(params.backbone)()
    body = get_model_class(params.model)(backbone, params)

    # Dataloaders
    # Note that both dataloaders sample identical episodes, via episode_seed
    support_epochs = n_epoch
    support_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=True,
                                                     n_query_shot=q, n_episodes=n_episodes, n_epochs=support_epochs,
                                                     augmentation=None, # params.ft_augmentation
                                                     unlabeled_ratio=0,
                                                     num_workers=params.num_workers,
                                                     split_seed=params.split_seed,
                                                     episode_seed=params.ft_episode_seed,
                                                     )

    query_loader = get_labeled_episodic_dataloader(params.target_dataset, n_way=w, n_shot=s, support=False,
                                                   n_query_shot=q, n_episodes=n_episodes, n_epochs=1,
                                                   augmentation=None,
                                                   unlabeled_ratio=0,
                                                   num_workers=params.num_workers,
                                                   split_seed=params.split_seed,
                                                   episode_seed=params.ft_episode_seed,
                                                   ) # eval_mode=params.ft_tta_mode)

    assert (len(support_loader) == n_episodes * support_epochs)
    assert (len(query_loader) == n_episodes)

    support_iterator = iter(support_loader)
    support_batches = math.ceil(n_data / bs)
    query_iterator = iter(query_loader)

    # Output (history, params)
    loss_history_path = get_ft_loss_history_path(output_dir)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    params_path = get_ft_params_path(output_dir)

    print('Saving finetune params to {}'.format(params_path))
    print('Saving finetune train history to {}'.format(train_history_path))
    if params.ft_tta_mode:
        test_history_path = test_history_path.replace(".csv", "_"+params.ft_tta_mode+".csv")
    print('Saving finetune test history to {}'.format(test_history_path))
    print()

    # saving parameters on this json file
    with open(params_path, 'w') as f_batch:
        json.dump(vars(params), f_batch, indent=4)
    
    # 저장할 dataframe
    df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_train = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)])
    
    # Pre-train state
    if not torch_pretrained:
        if params.source_dataset == 'miniImageNet':
            body_state_path = get_final_pretrain_state_path(base_output_dir)
        elif params.source_dataset == 'tieredImageNet':
            body_state_path = get_final_pretrain_state_path(base_output_dir)
            # body_state_path = './logs/baseline/output/pretrained_model/tiered/resnet18_base_LS_base/pretrain_state_0090.pt'
            
        if not os.path.exists(body_state_path):
            raise ValueError('Invalid pre-train state path: ' + body_state_path)

        print('Using pre-train state:', body_state_path)
        print()
        state = torch.load(body_state_path)
    else:
        pass

    # print time
    now = time.localtime()
    start = time.time()
    print("%02d/%02d %02d:%02d:%02d" %(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print()

    # For each episode
    for episode in range(n_episodes):
        # Reset models for each episode
        if not torch_pretrained:
            body.load_state_dict(copy.deepcopy(state), strict=False)  # note, override model.load_state_dict to change this behavior.
        else:
            body = get_model_class(params.model)(copy.deepcopy(backbone), params)

        head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features
        
        body.to(device)
        head.to(device)
        
        if not torch_pretrained:
            backbone_name = 'backbone.trunk'
        else:
            backbone_name = 'backbone'
                    
        criterion = nn.CrossEntropyLoss().to(device)

        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).to(device) # 각 요소를 반복 [000001111122222....]

        x_query = next(query_iterator)[0].to(device)
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).to(device)
        
        train_loss_history = []
        train_acc_history = []
        test_acc_history = []

        # For each epoch
        for epoch in range(n_epoch):
            if epoch < 50:
                opt_params = []
                opt_params.append({'params': body.parameters(), 'lr': params.body_lr, 'momentum' : 0.9, 'dampening' : 0.9, 'weight_decay' : 0.001})
                opt_params.append({'params': head.parameters(), 'lr': params.head_lr, 'momentum' : 0.9, 'dampening' : 0.9, 'weight_decay' : 0.001})
                optimizer = torch.optim.SGD(opt_params)
            else:
                opt_params = []
                opt_params.append({'params': body.parameters(), 'lr': 0, 'momentum' : 0.9, 'dampening' : 0.9, 'weight_decay' : 0.001})
                opt_params.append({'params': head.parameters(), 'lr': params.head_lr, 'momentum' : 0.9, 'dampening' : 0.9, 'weight_decay' : 0.001})
                optimizer = torch.optim.SGD(opt_params)                
            
            # Train
            if epoch < 50:
                body.train()
            else:
                body.eval()
            
            head.train()

            x_support = next(support_iterator)[0].to(device)

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s)

            # For each iteration
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]

                x_batch = x_support[batch_indices]
                y_batch = y_support[batch_indices]
                # f_batch = body_forward(x_batch, body, backbone, torch_pretrained, params)
                f_batch = body.backbone(x_batch)

                # head 거치기
                pred = head(f_batch)
                
                correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()
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
                    # f_query = body_forward(x_query, body, backbone, torch_pretrained, params)
                    f_query = body.backbone(x_query)
                    pred = head(f_query)
                    correct = torch.eq(y_query, pred.argmax(dim=1)).sum()
                test_acc = correct / pred.shape[0]
            else:
                test_acc = torch.tensor(0)
            
            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())

        df_loss.loc[episode + 1] = train_loss_history
        df_loss.to_csv(loss_history_path)
        df_train.loc[episode + 1] = train_acc_history
        df_train.to_csv(train_history_path)
        df_test.loc[episode + 1] = test_acc_history
        df_test.to_csv(test_history_path)

        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))
    end = time.time()

    print('Saved history to:')
    print(train_history_path)
    print(test_history_path)
    df_loss.to_csv(loss_history_path)
    df_train.to_csv(train_history_path)
    df_test.to_csv(test_history_path)
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
