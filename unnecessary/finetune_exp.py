import copy
import json
import math
import os
import pickle
import pandas as pd
import torch.nn as nn

from backbone import get_backbone_class
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
    output_dir = get_ft_output_directory(params, experiment=True)
    source_dir = get_ft_output_directory(params)
    print()
    print('Running fine-tune with output folder:')
    print(output_dir)
    print()

    # Settings
    n_episodes = 6
    bs = params.ft_batch_size
    n_data = params.n_way * params.n_shot
    n_epoch = int( math.ceil(n_data / 4) * params.ft_epochs / math.ceil(n_data / bs) )
    n_iter = int(n_epoch * math.ceil(n_data / bs))
    print("\nCurrent batch size:", bs)
    print("Current optimizer:", params.ft_optimizer)
    print("Current learning rate:", params.ft_lr)
    print("Currently updating :", params.ft_parts)
    print("Current scheduler :", params.ft_scheduler)
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

    support_batches = math.ceil(w * s / bs)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    loss_history_path = os.path.join(output_dir, 'loss_history.csv')
    grad_history_path = os.path.join(output_dir, 'grad_history.csv')
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
                            columns=[e + 1 for e in range(n_epoch)])
    df_test = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=[e + 1 for e in range(n_epoch)])

    # row : episode, col : iter
    df_loss = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=[e + 1 for e in range(n_iter)])
    df_grad = pd.DataFrame(None, index=list(range(1, n_episodes + 1)),
                            columns=[e + 1 for e in range(n_iter)])


    # Pre-train state
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

    # Loss function
    loss_fn = nn.CrossEntropyLoss().cuda()

    print('Starting fine-tune')
    if use_fixed_features:
        print('Running optimized fixed-feature fine-tuning (no augmentation, fixed body)')
    print()

    paths = []
    # sucess 3, fail 3
    for rank in range(1, 4):
        best_path = os.path.join(source_dir, 'best{}.txt'.format(rank))
        paths.append(best_path)
    for rank in range(1, 4):
        worst_path = os.path.join(source_dir, 'worst{}.txt'.format(rank))
        paths.append(worst_path)

    for episode in range(n_episodes):
        # Reset models for each episode
        body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
        head = get_classifier_head_class(params.ft_head)(body.final_feat_dim, params.n_way,
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
        optimizer = torch.optim.SGD(opt_params, lr=params.ft_lr, momentum=0.9, dampening=0.9, weight_decay=0.001)

        with open(paths[episode], 'rb') as f:
            test_acc_his = pickle.load(f)
            print("[{:.2f} accuracy] {}".format(float(test_acc_his), paths[episode][-9:-4]))
            x_support = pickle.load(f)
            y_support = pickle.load(f)
            x_query = pickle.load(f)
            y_query = pickle.load(f)

        f_support = None
        f_query = None

        with torch.no_grad():
            x_support = x_support.cuda()
            # body를 통과한 extracted feature
            # f_support shape : (25, 512)
            f_support = body.forward_features(x_support, params.ft_features)
            f_query = body.forward_features(x_query, params.ft_features)

        train_acc_history = []
        test_acc_history = []
        train_loss_history = []
        train_grad_history = []

        for epoch in range(n_epoch):
            # Train
            body.train()
            head.train()

            total_loss = 0
            correct = 0
            indices = np.random.permutation(w * s) # shuffle 5 * 5 or 1 * 5

            # iteration 25/bs(5shot) or 5/bs(1shot)
            for i in range(support_batches): # 5way : 7 / 1way : 2
                start_index = i * bs
                end_index = min(i * bs + bs, w * s)
                batch_indices = indices[start_index:end_index]
                y = y_support[batch_indices] # label of support set

                # body 거치기 
                if use_fixed_features:
                    f = f_support[batch_indices]
                else: # if use augmentation or update body
                    f = body.forward_features(x_support[batch_indices], params.ft_features)
                p = head(f)

                correct += torch.eq(y, p.argmax(dim=1)).sum()
                loss = loss_fn(p, y)

                optimizer.zero_grad() # pytorch에서는 이걸 안해주면 gradient를 계속 누적함 (각 Iteration이 끝나면 초기화해줌)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_loss_history.append(loss.item())

                train_grad = np.sqrt(sum(head.fc.weight.grad.cpu().flatten()**2)).item() #l2 norm of gradient
                train_grad_history.append(train_grad)

            train_loss = total_loss / support_batches
            train_acc = correct / (w * s)

            # Evaluation
            body.eval()
            head.eval()

            # Test Using Query
            if params.ft_intermediate_test or epoch == n_epoch - 1:
                with torch.no_grad():
                    p_query = head(f_query)
                test_acc = torch.eq(y_query, p_query.argmax(dim=1)).sum() / (w * q)
            else:
                test_acc = torch.tensor(0)

            print_epoch_logs = False
            if print_epoch_logs and (epoch + 1) % 10 == 0:
                fmt = 'Epoch {:03d}: Loss={:6.3f} Train ACC={:6.3f} Test ACC={:6.3f}'
                print(fmt.format(epoch + 1, train_loss, train_acc, test_acc))

            train_acc_history.append(train_acc.item())
            test_acc_history.append(test_acc.item())
            
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
        df_grad.loc[episode + 1] = train_grad_history
        df_grad.to_csv(grad_history_path)

        fmt = 'Episode {:03d}: train_loss={:6.4f} train_acc={:6.2f} test_acc={:6.2f}'
        print(fmt.format(episode, train_loss, train_acc_history[-1] * 100, test_acc_history[-1] * 100))

    fmt = 'Final Results: Acc={:5.2f} Std={:5.2f}'
    #print(fmt.format(df_test.mean()[-1] * 100, 1.96 * df_test.std()[-1] / np.sqrt(n_episodes) * 100))

    print('Saved history to:')
    print(train_history_path)
    print(test_history_path)
    df_train.to_csv(train_history_path)
    df_test.to_csv(test_history_path)
    df_loss.to_csv(loss_history_path)
    df_grad.to_csv(grad_history_path)


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
