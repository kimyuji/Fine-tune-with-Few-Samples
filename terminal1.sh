# gpu 1
# 이거 돌리고있드아!!!!! terminal 1

# for TARGET in "ChestX"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 
# done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 0 --ft_manifold_mixup --ft_augmentation randomhorizontalflip --ft_batch_size 8
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 0 --ft_manifold_mixup --ft_augmentation randomhorizontalflip --ft_batch_size 16
done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_cutmix both
# done

# for TARGET in "miniImageNet_test"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --ft_augmentation base --ft_label_smoothing 0.2
# done