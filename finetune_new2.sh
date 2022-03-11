# # 한번에 다돌려버리기~~ 

# # cutmix
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix v1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix v1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix v2
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix v2
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_cutmix v3
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix v3
# done

# # mixup
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup v1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup v1
# done

#여기부터
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup v2
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup v2
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup v2_reverse
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup v2_reverse
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_mixup v3
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup v3
done



# # manifold
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_manifold v1
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_manifold v1
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_manifold v2
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_manifold v2
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_manifold v3
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_manifold v3
# done


# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --ft_manifold mixup
# done

# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_manifold mixup
# done




# New arguments
# --startup_split: use 80% of data, as in Startup (default=False)
# --partial_reinit: reinit {Conv2, BN2, ShortCutConv, ShortCutBN} from last block (default=False)
# --mv_init: reinit all blocks with normal dist while maintaining mean-var
# --method startup_both_body
# --method startup_student_body
# --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

# python ./finetune.py --dataset miniImageNet --model ResNet10 --startup_split \
#  --method baseline_body --train_aug --reinit_blocks 1 2 3 4
# python ./finetune.py --dataset miniImageNet --model ResNet10 --startup_split \
#  --method baseline --train_aug --reinit_blocks 1 2 3 4

# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 5 --freeze_backbone --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 5 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --n_shot 5 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 1 --freeze_backbone --train_aug 
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 1 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --n_shot 1 --train_aug

### Pre-train
# python ./train_new.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug
# python ./train_new.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug