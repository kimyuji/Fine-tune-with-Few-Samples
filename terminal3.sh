# for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
#   python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model base --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --layer_diff
# done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score --ft_augmentation randomresizedcrop
done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --ft_augmentation randomresizedcrop
done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score --ft_cutmix both
done