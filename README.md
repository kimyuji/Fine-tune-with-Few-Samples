# How to Fine-tune Models with Few Samples : Update, Data Augmentation, and Test-time Augmentation

We used average of 2700MiB GPU memory on a single RTX 3090 per experiment, operated on Ubuntu. 
The code is implemented on Pytorch. 

To implement DA(data augmentation),
```
python ./finetune.py --ls --source_dataset miniImageNet --target_dataset miniImageNet_test --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_augmentation base
```

To implement DA+TTA(data augmentation for updating and test time augmentation for evaluation)
```
python ./finetune_da_tta.py --ls --source_dataset miniImageNet --target_dataset miniImageNet_test --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_augmentation base
```

The argument for applying update methods and augmentation techniques is as follows:

### Update Method
- LP : `--ft_parts head` <br>
- FT : `--ft_parts full`

### Single Augmentation

- Base Aug : `--ft_augmentation base` <br>
- RCrop : `--ft_augmentation rcrop` <br>
- HFlip : `--ft_augmentation hflip` <br>
- CJitter : `--ft_augmentation cjitter ` <br>


### Mixing Augmentation
- MixUp (W+B) : `--ft_mixup both` <br>
- CutMix (W+B) : `--ft_cutmix both` <br>
