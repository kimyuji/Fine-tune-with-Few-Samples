for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --ft_manifold_mixup --ft_train_with_clean --v_score
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --ft_manifold_mixup --ft_train_with_clean --v_score
done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --ft_augmentation base --ft_label_smoothing 0.2 --v_score
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --ft_augmentation base --ft_label_smoothing 0.2 --v_score
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_augmentation randomresizedcrop
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_augmentation randomcoloredjitter
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_augmentation randomhorizontalflip
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_mixup diff
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --ft_mixup same
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"  "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 1 --ft_batch_size 8 --v_score
# done



# for TARGET in "cars" "cub" "places" "plantae"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --ft_batch_size 16
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --layer_diff
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score --layer_diff
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_batch_size 16 
# done

