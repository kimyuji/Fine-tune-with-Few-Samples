import copy
import json
import math
import os
import pickle
import pandas as pd
import torch.nn as nn
#import torchsummary as summary

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

# output_dir : ./logs/output_bs3/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# base_output_dir : #./logs/output_baseline/mini/resnet10_simclr_LS_default/mini_test/05way_005shot_head_default
# 둘다 makedir true

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(params):
    # ft_scheduler configuration 

    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_idx
    print('Current cuda device:', torch.cuda.current_device())

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

    # 값이 맞게끔 보증! 
    assert (len(query_loader) == n_episodes)
    assert (len(support_loader) == n_episodes * support_epochs)

    query_iterator = iter(query_loader)
    support_iterator = iter(support_loader)
    support_batches = math.ceil(n_data / bs)

    # Output (history, params)
    train_history_path = get_ft_train_history_path(output_dir)
    test_history_path = get_ft_test_history_path(output_dir)
    loss_history_path = train_history_path.replace('train_history', 'loss_history')
    train_clean_history_path = test_history_path.replace('test_history', 'clean_history')
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

    for episode in range(n_episodes):
        # Reset models for each episode
        if not torch_pretrained:
            body.load_state_dict(copy.deepcopy(state))  # note, override model.load_state_dict to change this behavior.
        head = get_classifier_head_class(params.ft_head)(512, params.n_way,
                                                         params)  # TODO: apply ft_features
        body.cuda()
        head.cuda()
        
        body.eval()
        head.train()

        opt_params = []
        if params.ft_train_head:
            opt_params.append({'params': head.parameters()})
        if params.ft_train_body:
            opt_params.append({'params': body.parameters()})

        # Optimizer and Learning Rate Scheduler
        # select optimizer
        optimizer = torch.optim.SGD(opt_params, lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
            
        # Loss function
        criterion = nn.CrossEntropyLoss().cuda()

        # no augmentation (Transform X)
            # x_support, y_support : episode마다 --> 절대 바뀌면 안됨
            # x_support_aug : epoch마다 --> based on x_support
            # y_shuffled : y_support와 pair
        
        # yes augmentation (augmentation O)
            # x_support, y_support : epoch마다 --> 절대 바뀌면 안됨
            # x_support_aug : epoch마다 dataloader통해
            # y_shuffled: 없음 X

        x_support = None
        f_support = None
        y_support = torch.arange(w).repeat_interleave(s).cuda() # 각 요소를 반복 [000001111122222....]

        x_query = next(query_iterator)[0].cuda()
        f_query = None
        y_query = torch.arange(w).repeat_interleave(q).cuda() 

        with torch.no_grad():
            if torch_pretrained:
                f_query = backbone(x_query)
            else:
                f_query = body.forward_features(x_query, params.ft_features)
            # load support set
            if use_fixed_features:  # load data and extract features once per episode
                x_support, _ = next(support_iterator)
                x_support = x_support.cuda()

                if torch_pretrained:
                    f_support_clean = backbone(x_support)
                else:
                    f_support_clean = body.forward_features(x_support, params.ft_features)

        train_acc_history = []
        test_acc_history = []
        train_acc_clean_history = []
        train_loss_history = []

        for epoch in range(n_epoch):
            # 4 augmentation methods : mixup, cutmix, manifold, augmentation(transform)
            # mixup, cutmix, manifold mixup need 2 labels <- mix_bool == True
            mix_bool = (params.ft_mixup or params.ft_cutmix or params.ft_manifold_mixup) 
            aug_bool = mix_bool or params.ft_augmentation
            if params.ft_scheduler_end is not None: # if aug is scheduled, 
                aug_bool = (epoch < params.ft_scheduler_end and epoch >= params.ft_scheduler_start) and aug_bool

            if not use_fixed_features:
                x_support_aug, _ = next(support_iterator) # augmented by loader
                x_support_aug = x_support_aug.cuda() 
                    

            total_loss = 0
            correct = 0
            indices = np.random.permutation(n_data) # for shuffling


            # img cutmix & img mixup & manifold mixup
            if aug_bool:
                if mix_bool:
                    x_support_aug = copy.deepcopy(x_support)
                    lam = np.random.beta(1.0, 1.0) 
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x_support.shape, lam)
                    indices_shuffled = torch.randperm(x_support.shape[0])
                    y_shuffled = y_support[indices_shuffled] 

                    if params.ft_cutmix: 
                        x_support_aug[:,:,bbx1:bbx2, bby1:bby2] = x_support[indices_shuffled,:,bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_support.shape[-1] * x_support.shape[-2])) # adjust lambda
                    elif params.ft_mixup:
                        x_support_aug = lam * x_support[:,:,:] + (1. - lam) * x_support[indices_shuffled,:,:]

                with torch.no_grad():
                    if torch_pretrained:
                        f_support = backbone(x_support_aug).squeeze(-1).squeeze(-1)
                    else:
                        f_support = body.forward_features(x_support_aug, params.ft_features)
            else : 
                f_support = f_support_clean

            # iteration 25/bs(5shot) or 5/bs(1shot)
            for i in range(support_batches):
                start_index = i * bs
                end_index = min(i * bs + bs, n_data)
                batch_indices = indices[start_index:end_index]

                f_batch = f_support[batch_indices]
                y_batch = y_support[batch_indices] # label of support set

                if aug_bool and mix_bool:
                    y_shuffled_batch = y_shuffled[batch_indices]
                
                # head 거치기
                pred = head(f_batch)

                correct += torch.eq(y_batch, pred.argmax(dim=1)).sum()

                if aug_bool and mix_bool:
                    loss = criterion(pred, y_batch) * lam + criterion(pred, y_shuffled_batch) * (1. - lam)
                else:
                    loss = criterion(pred, y_batch)

                optimizer.zero_grad() # pytorch에서는 이걸 안해주면 gradient를 계속 누적함 (각 Iteration이 끝나면 초기화해줌)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if params.ft_lr_scheduler:
                scheduler.step()
            train_loss = total_loss / support_batches
            train_acc = correct / n_data

            # Evaluation
            body.eval()
            head.eval()

            # Test Using Query
            if params.ft_intermediate_test or epoch == n_epoch - 1:
                with torch.no_grad():
                    if torch_pretrained:
                        f_query = f_query.squeeze(-1).squeeze(-1)
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
