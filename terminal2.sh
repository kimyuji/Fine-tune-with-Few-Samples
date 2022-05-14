for TARGET in "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --ft_augmentation randomcoloredjitter --ft_batch_size 16
done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --ft_mixup both --ft_scheduler_start 0 --ft_scheduler_end 70
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --ft_mixup both --ft_scheduler_start 30 --ft_scheduler_end 100
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score --ft_mixup both --ft_scheduler_start 0 --ft_scheduler_end 70
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score --ft_mixup both --ft_scheduler_start 30 --ft_scheduler_end 100
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

