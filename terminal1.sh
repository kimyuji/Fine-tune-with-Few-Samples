# aug base

for TARGET in "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_augmentation base --ft_batch_size 16
done
