# python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset miniImageNet_test --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix  --ft_clean_test
# done



# # 한번에 다돌려버리기~~ 


# # manifold v2 + cutmix v3
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix v3 --ft_manifold v2
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix v3 --ft_manifold v2
# done


# # middle
# for TARGET in "miniImageNet_test" "CropDisease"  "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation base --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix v3 --ft_manifold v2
# done


# aug 
# for TARGET in "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation randomcoloredjitter --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation randomgaussianblur --ft_clean_test
# done


# for TARGET in  "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation randomresizedcrop --ft_clean_test 
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation randomresizedcrop --ft_clean_test 
# done

# for TARGET in  "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_augmentation randomhorizontalflip --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation randomhorizontalflip --ft_clean_test 
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_parts full
# done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_parts full
done


# middle

# # 20 80
# # cutmix

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix --ft_scheduler_end 70 --ft_scheduler_start 30 --ft_clean_test
# done




# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_scheduler_end 80 --ft_scheduler_start 20 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup --ft_scheduler_end 80 --ft_scheduler_start 20 --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_scheduler_end 90 --ft_scheduler_start 10 --ft_clean_test
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new2.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup --ft_scheduler_end 90 --ft_scheduler_start 10 --ft_clean_test
# done



