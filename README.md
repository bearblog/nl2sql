## version0630 

*  官方baseline,sqlnet_condition_column_cat_attention_300dimtanh
Dev Logic Form Accuracy: 0.363, Execution Accuracy: 0.472   0.75113
*   官方baseline,sqlnet_condition_column_cat_attention_100dimtanh
Dev Logic Form Accuracy: 0.324, Execution Accuracy: 0.446   0.72133
*   官方baseline,sqlnet_condition_column_cat_attention_300dim
Dev Logic Form Accuracy: 0.361, Execution Accuracy: 0.484   0.75159
*官方baseline,sqlnet_condition_column_cat_attention_100dim   
Dev Logic Form Accuracy: 0.317, Execution Accuracy: 0.443   0.7099

## version0621
官方baseline , 尝试跑通 + 重构(bz=16)

Dev Logic Form Accuracy: 0.312, Execution Accuracy: 0.439 onLine: 0.3694


## Experiment analysis

We found the main challenges of this datasets containing poor condition value prediction, select column and condition column not mentioned in NL question, inconsistent condition relationship representation between NL question and SQL, etc. All these challenges could not be solved by existing baseline and SOTA models.

Correspondingly, this baseline model achieves only 77% accuracy on condition column and 62% accuracy on condition value respectively even on the training set, and the overall logic form is only around 50% as well, indicating these problems are challenging for contestants to solve.

<div align="middle"><img src="https://github.com/ZhuiyiTechnology/nl2sql_baseline/blob/master/img/trainset_behavior.png"width="80%" ></div>

## Related resources:
https://github.com/salesforce/WikiSQL

https://yale-lily.github.io/spider

<a href="https://arxiv.org/pdf/1804.08338.pdf">Semantic Parsing with Syntax- and Table-Aware SQL Generation</a>
