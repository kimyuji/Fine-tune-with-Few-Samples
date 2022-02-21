import copy
import json
import math
import os
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
import torchsummary as summary
import torchvision
from tqdm import tqdm

from backbone import get_backbone_class
import backbone
from datasets.dataloader import get_labeled_episodic_dataloader
from io_utils import parse_args
from model import get_model_class
from model.classifier_head import get_classifier_head_class
from paths import get_output_directory, get_ft_output_directory, get_ft_train_history_path, get_ft_test_history_path, \
    get_final_pretrain_state_path, get_pretrain_state_path, get_ft_params_path
from utils import *

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

    w = params.n_way
    s = params.n_shot
    q = params.n_query_shot

    

    # Settings
    n_episodes = 1
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot
    n_epoch = 100

    df_test_history = pd.DataFrame(None, index=list(range(0, 100)), # 10의자리 : support index, 1의자리 : query index
                           columns=['epoch{}'.format(e + 1) for e in range(n_epoch)]) # for showing training procedure
    df_test_episode = pd.DataFrame(None, index=['support_{}'.format(e) for e in range(10)], # index : support index, column : query index
                           columns=['query_{}'.format(e) for e in range(10)])
    
    test_history_path = os.path.join(output_dir, "sq_test_history.csv")
    test_episode_path = os.path.join(output_dir, "sq_test_acc.csv")
    print()
    
    # Whether to optimize for fixed features (when there is no augmentation and only head is updated)
    use_fixed_features = params.ft_augmentation is None and params.ft_parts == 'head'
    support_batches = math.ceil(w * s / bs)

    head = get_classifier_head_class(params.ft_head)(512, params.n_way, params)  # TODO: apply ft_features
    head.train()
    head.cuda()

    opt_params = []
    opt_params.append({'params': head.parameters()})
    optimizer = torch.optim.SGD(opt_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss().cuda()       

    for support_idx in range(10):
        f_support = np.load(os.path.join(output_dir, 'embedding/{}_support.npy'.format(support_idx)))
        print(os.path.join(output_dir, 'embedding/{}_support.npy'.format(support_idx)))
        f_support = torch.Tensor(f_support).squeeze(-1).squeeze(-1).cuda()
        for query_idx in range(10):
            test_acc_history = []
            f_query = np.load(os.path.join(output_dir, 'embedding/{}_query.npy'.format(query_idx)))
            f_query = torch.Tensor(f_query).squeeze(-1).squeeze(-1).cuda()

            for epoch in range(n_epoch):
                y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]
                y_query = torch.arange(w).repeat_interleave(q).cuda()
                indices = np.random.permutation(w * s)

                total_loss = 0
                correct = 0
                for i in range(support_batches):
                    start_index = i * bs
                    end_index = min(i * bs + bs, w * s)
                    
                    batch_indices = indices[start_index:end_index]
                    y = y_support[batch_indices] # label of support set
                    f = f_support[batch_indices]
                
                    # head 거치기
                    p = head(f)

                    correct += torch.eq(y, p.argmax(dim=1)).sum()
                    loss = loss_fn(p, y)

                    optimizer.zero_grad() # pytorch에서는 이걸 안해주면 gradient를 계속 누적함 (각 Iteration이 끝나면 초기화해줌)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                train_loss = total_loss / support_batches
                train_acc = correct / (w * s)

                # Evaluation
                head.eval()

                with torch.no_grad():
                    p_query = head(f_query)
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)

                test_acc_history.append(test_acc.item())

            df_index = support_idx * 10 + query_idx
            df_test_history.loc[df_index] = test_acc_history
            fmt = 'Episode {:03d}: train_loss={:6.4f} test_acc={:6.2f}'
            print(fmt.format(df_index, train_loss, test_acc_history[-1] * 100))
            df_test_episode.iloc[support_idx][query_idx] = test_acc_history[-1]

        df_test_history.to_csv(test_history_path)
        

    df_test_history.to_csv(test_history_path)
    df_test_episode.to_csv(test_episode_path)



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
