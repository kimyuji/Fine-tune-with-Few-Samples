# TTA ver.3 
for TARGET in "ISIC" ; do
  python ./finetune_full.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts full --split_seed 1 --n_shot 5 --gpu_idx 0 --v_score --ft_mixup both --ft_intermediate_test
done