# GPU 1

# Baseline shots 

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_clean_test --ft_batch_size 16 --ft_cutmix
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 50 --ft_clean_test --ft_batch_size 32 --ft_cutmix
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 20 --ft_clean_test --ft_batch_size 16 --ft_mixup
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_head.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 50 --ft_clean_test --ft_batch_size 32 --ft_mixup
done