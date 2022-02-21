import glob
import os
import re
from argparse import Namespace

import configs

DATASET_KEYS = {
    'miniImageNet': 'mini',
    'miniImageNet_test': 'mini_test',
    'tieredImageNet': 'tiered',
    'tieredImageNet_test': 'tiered_test',
    'CropDisease': 'crop',
    'EuroSAT': 'euro',
    'ISIC': 'isic',
    'ChestX': 'chest',
    'cars': 'cars',
    'cub': 'cub',
    'places': 'places',
    'plantae': 'plantae',
}

BACKBONE_KEYS = {
    'resnet10': 'resnet10',
    'resnet18': 'resnet18',
    'torch_resnet18' : 'torch_resnet18',
    'torch_resnet34' : 'torch_resnet34',
    'torch_resnet50' : 'torch_resnet50',
    'torch_resnet101' : 'torch_resnet101',
    'torch_resnet152' : 'torch_resnet152',
}

MODEL_KEYS = {
    'base': 'base',
    'simclr': 'simclr',
    'simsiam': 'simsiam',
    'moco': 'moco',
    'swav': 'swav',
    'byol': 'byol',
}


def get_output_directory(params: Namespace, pls_previous=False, makedirs=True):
    # python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset ChestX --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
    # ./logs/output/mini/resnet10_simclr_UT_default/chest/05way_005shot_head_default
    """
    :param params:
    :param pls_previous: get previous output directory for pls mode
    :return:
    """
    if pls_previous and not params.pls:
        raise Exception('Should not get pls_previous when params.pls is False')

    path = configs.save_dir
    path = os.path.join(path, 'baseline')
    path = os.path.join(path, 'output')
    # path = os.path.join(path, DATASET_KEYS[params.source_dataset])

    pretrain_specifiers = []
    pretrain_specifiers.append(BACKBONE_KEYS[params.backbone])
    if pls_previous:
        pretrain_specifiers.append(MODEL_KEYS['base'])
        pretrain_specifiers.append('LS')
        pretrain_specifiers.append(params.pls_tag)
    else:
        pretrain_specifiers.append(MODEL_KEYS[params.model])
        if params.pls:
            pretrain_specifiers.append('PLS')
        if params.ls:
            pretrain_specifiers.append('LS')
        if params.us:
            pretrain_specifiers.append('US')
        if params.ut:
            pretrain_specifiers.append('UT')
        pretrain_specifiers.append(params.tag)
    path = os.path.join(path, '_'.join(pretrain_specifiers))

    if params.ut and not pls_previous:
        path = os.path.join(path, DATASET_KEYS[params.target_dataset])

    if makedirs:
        os.makedirs(path, exist_ok=True)

    return path 


def get_pretrain_history_path(output_directory):
    basename = 'pretrain_history.csv'
    return os.path.join(output_directory, basename)


def get_pretrain_state_path(output_directory, epoch=0):
    """
    :param output_directory:
    :param epoch: Number of completed epochs. I.e., 0 = initial.
    :return:
    """
    basename = 'pretrain_state_{:04d}.pt'.format(epoch)
    return os.path.join(output_directory, basename)


def get_final_pretrain_state_path(output_directory):
    glob_pattern = os.path.join(output_directory, 'pretrain_state_*.pt')
    paths = glob.glob(glob_pattern)

    pattern = re.compile('pretrain_state_(\d{4}).pt')
    paths_by_epoch = dict()
    for path in paths:
        match = pattern.search(path)
        if match:
            paths_by_epoch[match.group(1)] = path

    if len(paths_by_epoch) == 0:
        return " "
        raise FileNotFoundError('Could not find valid pre-train state file in {}'.format(output_directory))

    max_epoch = max(paths_by_epoch.keys())
    return paths_by_epoch[max_epoch]


def get_pretrain_params_path(output_directory):
    return os.path.join(output_directory, 'pretrain_params.json')


def get_ft_output_directory(params, makedirs=True, experiment=False):
    path = get_output_directory(params, makedirs=makedirs)
    if params.ft_batch_size != 4:
        path = get_output_directory(params, makedirs=makedirs).replace("output", "output_{:03d}".format(params.ft_batch_size))
        path = path.replace("baseline", "batch_size")
    # if params.ft_parts != 'head':
    #     path = get_output_directory(params, makedirs=makedirs).replace("baseline", "part_{}".format(params.ft_parts))
    if params.ft_optimizer != 'SGD':
        path = get_output_directory(params, makedirs=makedirs).replace("output", "output_{}".format(params.ft_optimizer))
        path = path.replace("baseline", "optimizer")
    if params.ft_lr != 0.01 :
        path = get_output_directory(params, makedirs=makedirs).replace("output", "output_{}".format(params.ft_lr))
        path = path.replace("baseline", "learning_rate")
    if params.ft_scheduler != 'None':
        path = get_output_directory(params, makedirs=makedirs).replace("output", "output_{}".format(params.ft_scheduler))
        path = path.replace("baseline", "lr_scheduler")        

    if not params.ut:
        path = os.path.join(path, DATASET_KEYS[params.target_dataset])
    ft_basename = '{:02d}way_{:03d}shot_{}_{}'.format(params.n_way, params.n_shot, params.ft_parts, params.ft_tag)
    path = os.path.join(path, ft_basename)

    if experiment == True:
        path = path.replace("baseline/output", "experiment/perplexity")

    if makedirs:
        os.makedirs(path, exist_ok=True)
    return path


def get_ft_params_path(output_directory):
    return os.path.join(output_directory, 'params.json')


def get_ft_train_history_path(output_directory):
    return os.path.join(output_directory, 'train_history.csv')


def get_ft_test_history_path(output_directory):
    return os.path.join(output_directory, 'test_history.csv')