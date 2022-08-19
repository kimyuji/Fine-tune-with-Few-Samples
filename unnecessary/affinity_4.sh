
for TARGET in "plantae" "tieredImageNet_test"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 1 --gpu_idx 3
done

for TARGET in "plantae" "tieredImageNet_test"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 5 --gpu_idx 3 
done

for TARGET in "plantae" "tieredImageNet_test"; do
  python ./finetune_scratch.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr --ft_parts scratch --split_seed 1 --n_shot 20 --gpu_idx 3 --ft_batch_size 16
done

