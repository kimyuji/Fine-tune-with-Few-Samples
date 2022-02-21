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