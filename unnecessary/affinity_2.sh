
for TARGET in "ISIC" "ChestX"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 1 --gpu_idx 1 
done

for TARGET in "ISIC" "ChestX"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 5 --gpu_idx 1 
done

for TARGET in "ISIC" "ChestX"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 20 --gpu_idx 1 --ft_batch_size 16
done

