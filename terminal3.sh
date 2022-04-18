for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./extract_feat.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./extract_feat.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5
done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts full --split_seed 1 --ft_intermediate_test --ft_cutmix --ft_clean_test 
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_clean_test
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts full --split_seed 1 --ft_intermediate_test --ft_mixup --ft_clean_test 
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT"; do
#   python ./finetune.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts full --split_seed 1 --ft_intermediate_test  
# done
