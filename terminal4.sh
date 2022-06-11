## Only for 10 shot experiments!!!


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 3 --v_score --ft_batch_size 8
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts head --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 3 --ft_batch_size 8
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 3 --v_score --ft_batch_size 8 --ft_mixup both 
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 3 --ft_batch_size 8 --ft_augmentation randomcoloredjitter
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 10 --gpu_idx 3 --ft_batch_size 8 --ft_augmentation randomhorizontalflip
done
