# valid : fixed aug, tta : fixed aug

for TARGET in "miniImageNet_test" ; do
  python ./finetune_shift.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_augmentation base
done
