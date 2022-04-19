for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_aug.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix 
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_aug.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix
done


# PLS는 일단 쓰지 말라 하셨음! 
## Type 1 # labeled source 
# python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet18 --model simclr --epochs 1000

## Type 2 
# python ./pretrain_new.py --ut --source_dataset miniImageNet --backbone resnet10 --model simclr --epochs 200

# ## Type 3 #unlabeled target 
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./pretrain_new.py --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --augmentation base
# done

# ## PLS pretrain (note 1, should use --model base for ls pretrain) (note 2, should use same --tag for ls and pls)
# python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet10 --model base

# ## Type 4
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./pretrain_new.py --pls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
# done

# # Type 5
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./pretrain_new.py --pls --ls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
# done
