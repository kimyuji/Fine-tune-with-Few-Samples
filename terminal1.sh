# gpu 1
# 이거 돌리고있드아!!!!! terminal 1


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_clean_test --ft_mixup --gpu_idx 1 --ft_batch_size 16
done


for TARGET in "ChestX"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_clean_test --ft_cutmix --gpu_idx 1 --ft_batch_size 16
done


# # Flip 

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation randomhorizontalflip --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_augmentation randomhorizontalflip --ft_clean_test
# done

# # Resized Crop

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_augmentation randomresizedcrop --ft_clean_test
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_augmentation randomresizedcrop --ft_clean_test
# done