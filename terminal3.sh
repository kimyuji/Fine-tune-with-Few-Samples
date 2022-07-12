for TARGET in "miniImageNet_test" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --ft_intermediate_test --gpu_idx 2 --ft_augmentation randomhorizontalflip --ft_tta_mode fixed_hflip --ft_valid_mode clean --ft_intermediate_test
done
