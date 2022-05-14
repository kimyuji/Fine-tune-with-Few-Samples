# gpu 1
# 이거 돌리고있드아!!!!! terminal 1

# for TARGET in "miniImageNet_test" "tieredImageNet_test"; do
#   python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --ft_update_scheduler LP-FT
# done

for TARGET in "ISIC"; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --ft_intermediate_test --n_shot 1 --gpu_idx 1 --ft_update_scheduler LP-FT
done