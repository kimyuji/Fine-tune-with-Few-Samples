# make sure to set "--gpu_idx 1"

for TARGET in "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --ft_batch_size 16
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --layer_diff
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score --layer_diff
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_batch_size 16 --layer_diff
done



# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./get_0_epoch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --ft_augmentation base --n_shot 1 --gpu_idx 1 --v_score
# done 

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./get_0_epoch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --ft_augmentation base --n_shot 5 --gpu_idx 1 --v_score
# done 

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

