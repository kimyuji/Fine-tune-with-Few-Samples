# early stopping

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 1 --ft_intermediate_test --gpu_idx 0 
done

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone torch_resnet18 --model simclr --ft_parts head --split_seed 1 --n_shot 5 --ft_intermediate_test --gpu_idx 0 
done