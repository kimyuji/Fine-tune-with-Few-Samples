# Instability of Fine-TUning 


## Jupyter Notebook File info 


- analysis_domain_diff.ipynb
calculate domain difference/similarity across datasets <br>
[Large scale fine-grained categorization and domain-specific transfer learning](https://arxiv.org/pdf/1806.06193.pdf) <br>
[source code](https://github.com/richardaecn/cvpr18-inaturalist-transfer/blob/master/DomainSimilarityDemo.ipynb)
- analysis_img.ipynb
check imgs & calculate img similarities btw imgs of size (3, 224, 224)

- analysis_instab.ipynb
calculate instability measures <br>
std() or std()/mean()

- analysis_ordering.ipynb
visualize ordering experiment results

- analysis_support_query.ipynb
    calculate clustering measure & hardness of episode
    - clustering measure : [Unraveling Meta-Learning: Understanding Feature
    Representations for Few-Shot Tasks](http://proceedings.mlr.press/v119/goldblum20a/goldblum20a.pdf)
    - hardness : [A baseline for few-shot image classification](https://arxiv.org/pdf/1909.02729.pdf)<br>
        [source code](https://github.com/kimyuji/few-shot-baseline/blob/bfd77ddc65fe4e5e70789fd5751f7337e18c4cd6/utils.py#L197)
        
## Additional Python File(.py) info
- extract_feat.py
  - main() : extract feature by resnet10 (pretrained on mini-imagenet) that we pretrained 
      - output directory : './feature/{dataset name}/'
  - main2() : extract feature by resnet101 (pretrained on imagenet)
      - output directory : './logs/img_ft_difference/feature/{dataset name}/resnet101/{}shot/'
  
- sample_sq.py
    for check support, query relationship by different episodes, I sampled 10 independent support and query sets respectively, and extract features by resnet152 and resnet18
    - resnet 152 (feat_extract object) : for calculating clustering measure & hardness (will be used in analysis_support_query.ipynb)
        - output directory : './logs/baseline/output/torch_resnet18_simclr_LS_default/{dataset_name}/05way_00{}shot_head_default/feature/'
    - resnet18 (backbone object) : for calcating accuracy (will be used in evaluate_sq.py)
        - output directory : './logs/baseline/output/torch_resnet18_simclr_LS_default/{dataset_name}/05way_00{}shot_head_default/embedding/'
    **HAVE TO SET ARGUMENT AS --backbone torch_resnet18**

- evaluate_sq.py
    for 100 different combinations of episodes using 10 support and query sets, we train on support set and evaluate on query set
    - output file (dataframe of accuracy) : './logs/baseline/output/torch_resnet18_simclr_LS_default/{dataset_name}/05way_00{}shot_head_default/sq_test_acc.csv'


- make_init.py
produce different random initialization and save (used in finetune_init.py) 
- finetune_init.py
for initialization experiment
- finetune_order.py
for ordering experiment
