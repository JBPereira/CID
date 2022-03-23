# CID
Covered Information Disentanglement (CID) corrects the permutation importance bias in the presence of covariates by using a map between permutation importance values and uncovered feature information values.


Run MV_comparison_CID.py for a demo showcasing how permutation importance is biased while CID can recover the right feature importance when there is high multicollinearity between the features. 

For a generated multivariate non-normal distribution with true importance I_1>I_2>I_3>I_4>I_5=I_6>I_7 this is the comparison of CID, permutation importance and gini importance:

![MV_non_normal_CID_comparison](https://github.com/JBPereira/CID/blob/main/plots/MV_non_normal_CID_comparison.png)

