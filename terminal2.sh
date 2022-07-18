# aug base, tta base

for TARGET in "miniImageNet_test"  "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_2_stage_reg.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 1 --ft_augmentation base --ft_tta_mode fixed_aug
done

for TARGET in "miniImageNet_test"  "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_2_stage_reg.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 1 --ft_augmentation base --ft_tta_mode fixed_aug
done
