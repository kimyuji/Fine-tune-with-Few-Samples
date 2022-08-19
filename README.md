# How to Fine-tune Models with Few Samples : Update, Data Augmentation, and Test-time Augmentation

We used a single RTX 3090 with average of 2700MiB GPU memory, operated on Ubuntu. 
The code is implemented on Pytorch. 

### Single Augmentation
--ft_augmentation base
--ft_augmentation rcrop
--ft_augmentation hflip
--ft_augmentation cjitter 

### Mixing Augmentation
--ft_mixup both
--ft_cutmix both

