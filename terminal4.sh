# gpu 1 
# 잘못된 코드지만 혹시라도 성능이 잘 나올까 하여... 

# cutmix
# 30-70
# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
#   python ./finetune_new_full_check.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_cutmix --ft_clean_test --gpu_idx 1 --ft_scheduler_start 30 --ft_scheduler_end 70
# done
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_full_check.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_cutmix --ft_clean_test --gpu_idx 1 --ft_scheduler_start 30 --ft_scheduler_end 70
done



# mixup
# 30-70
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_full_check.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 1 --ft_mixup --ft_clean_test --gpu_idx 1 --ft_scheduler_start 30 --ft_scheduler_end 70
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new_full_check.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 5 --ft_mixup --ft_clean_test --gpu_idx 1 --ft_scheduler_start 30 --ft_scheduler_end 70
done

