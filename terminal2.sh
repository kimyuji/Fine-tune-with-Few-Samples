# make sure to set "--gpu_idx 1"


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1
# done


# for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_raw_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_augmentation randomhorizontalflip --ft_batch_size 16 --gpu_idx 1
# done

for TARGET in "ChestX"; do
  python ./finetune_raw_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_augmentation randomhorizontalflip --ft_batch_size 16 --gpu_idx 1
done 

for TARGET in "ISIC"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_cutmix --ft_batch_size 16 --gpu_idx 1
done 

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT"; do
#   python ./finetune_raw_old.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_augmentation randomhorizontalflip --ft_batch_size 16
# done 



# for TARGET in "ChestX"; do
#   python ./finetune_new_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1 --ft_batch_size 16
# done


# # Resized Crop + Flip

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1 --ft_batch_size 16
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 50 --ft_augmentation flipcrop --ft_clean_test --gpu_idx 1 --ft_batch_size 32
# done

