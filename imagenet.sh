# source dataset 아무거나 해도 backbone을 torch_resnet18로 해두면 알아서 torch pretrained model 불러옵니당

### LP-FT Experiment
# Full
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc aug --v_score
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc aug --v_score
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc aug --v_score
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc aug --v_score
done

# Head
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 1 --gpu_idx 0
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 5 --gpu_idx 0
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16
done

### DA Experiment
# HFlip
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomhorizontalflip
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomhorizontalflip
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_augmentation randomhorizontalflip
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_augmentation randomhorizontalflip
done

# RCrop
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomresizedcrop
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomresizedcrop
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_augmentation randomresizedcrop
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_augmentation randomresizedcrop
done

# CJitter
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomcoloredjitter
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImage Net_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation randomcoloredjitter
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_augmentation randomcoloredjitter
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_augmentation randomcoloredjitter
done

# Base Aug.
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation base
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_augmentation base
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_augmentation base
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_augmentation base
done

# MixUp
# (Both)
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_mixup both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_mixup both
done

# (Same)
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup same
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup same
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_mixup same
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_mixup same
done

# (Diff)
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup diff
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_mixup diff
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_mixup diff
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_mixup diff
done

# CutMix
# (Both)
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_cutmix both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --ft_valid_acc clean --v_score --ft_cutmix both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 10 --gpu_idx 0 --ft_batch_size 8 --ft_valid_acc clean --v_score --ft_cutmix both
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts full --split_seed 1 --n_shot 20 --gpu_idx 0 --ft_batch_size 16 --ft_valid_acc clean --v_score --ft_cutmix both
done
