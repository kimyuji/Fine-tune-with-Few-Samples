for TARGET in "miniImageNet_test"  "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_1_stage_reg.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 5 --ft_mixup both --one_stage_reg both_CE
done

for TARGET in "miniImageNet_test"  "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_1_stage_reg.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 5 --ft_mixup both --one_stage_reg both_CE
done