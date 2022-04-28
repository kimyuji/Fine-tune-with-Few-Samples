# gpu 1 

for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX" ; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --v_score
done


for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_full.py --ls --source_dataset tieredImageNet --target_dataset $TARGET --backbone resnet18 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 5 --gpu_idx 1 --v_score
done