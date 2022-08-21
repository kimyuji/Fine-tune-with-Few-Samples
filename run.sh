# For analysis (full fine-tuning vs. linear probing)
for TARGET in "EuroSAT"; do
    python ./finetune.py --ls --source_dataset miniImageNet --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --target_dataset $TARGET
done

# For Original / TTA
for TARGET in "EuroSAT"; do
    python ./finetune_da_tta.py --ls --source_dataset miniImageNet --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --target_dataset $TARGET
done

# For DA / DA+TTA
for TARGET in "EuroSAT"; do
    python ./finetune_da_tta.py --ls --source_dataset miniImageNet --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 1 --gpu_idx 0 --ft_augmentation base --target_dataset $TARGET
done
