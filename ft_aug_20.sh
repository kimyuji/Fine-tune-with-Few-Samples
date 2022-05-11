for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_augmentation randomhorizontalflip --ft_batch_size 16
done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_augmentation randomresizedcrop --ft_batch_size 16
done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_augmentation base --ft_batch_size 16
done

for TARGET in "tieredImageNet_test" "cars" "cub" "places" "plantae"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 20 --gpu_idx 1 --v_score --ft_cutmix both --ft_batch_size 16
done

