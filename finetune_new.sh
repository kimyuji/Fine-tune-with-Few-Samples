for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./fine.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./fine.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./fine.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./fine.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
done


# for TARGET in "miniImageNet_test" ; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1  --n_shot --ft_intermediate_test
# done

# for TARGET in "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1  --n_shot 1 --ft_intermediate_test
# done

# for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done


# for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2_real.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done