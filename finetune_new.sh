# python ./evaluate_sq.py --ls --source_dataset miniImageNet --target_dataset miniImageNet_test --backbone torch_resnet18 --model simclr --split_seed 1 --ft_intermediate_test


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./evaluate_sq.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --split_seed 1 --ft_intermediate_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./evaluate_sq.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --split_seed 1  --n_shot 1 --ft_intermediate_test
# done

# baseline
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_clean_test
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_clean_test
done

# reverse
for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_scheduler_end 100 --ft_scheduler_start 70 --ft_clean_test
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_scheduler_end 100 --ft_scheduler_start 70 --ft_clean_test
done

# front
for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_scheduler_end 100 --ft_scheduler_start 70 --ft_clean_test
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_scheduler_end 30 --ft_scheduler_start 0 --ft_clean_test
done

# middle
for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done