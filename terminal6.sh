
# manifold mixup scheduler

# 20 shot
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --ft_intermediate_test --gpu_idx 0  --ft_manifold_mixup both --ft_scheduler_start 30 --ft_scheduler_end 70 --ft_batch_size 16
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --ft_intermediate_test --gpu_idx 0  --ft_manifold_mixup both --ft_scheduler_start 0 --ft_scheduler_end 70 --ft_batch_size 16
done
