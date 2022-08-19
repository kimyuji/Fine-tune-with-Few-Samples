# How to Fine-tune Models with Few Samples : Update, Data Augmentation, and Test-time Augmentation

We used average of 2700MiB GPU memory on a single RTX 3090 per experiment, operated on Ubuntu. 
The code is implemented on Pytorch. 

```
python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset miniImageNet_test --backbone resnet10 --model base --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_augmentation base
```

The argument for applying augmentation is as follows: 

### Single Augmentation
```
--ft_augmentation base
--ft_augmentation rcrop
--ft_augmentation hflip
--ft_augmentation cjitter 
```

### Mixing Augmentation
```
--ft_mixup both
--ft_cutmix both
```