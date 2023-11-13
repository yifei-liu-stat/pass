# PASS: Perturbation-Assisted Sample Synthesis
[Arxiv](https://arxiv.org/abs/2305.18671) | [BibTex](#bibtex)

Code repo for "PASS: Perturbation Assisted Sample Synthesis --- A Novel Approach For Uncertainty Quantification" (Y Liu, R Shen, X Shen).


<p align="center">
<img src=reports/figures/flowchart_pass_v2.png />
</p>

PASS is a generalization to proposed methods in [Shen et al., 2022](https://hdsr.mitpress.mit.edu/pub/x1ozqj10/release/3) and [Bi and Shen, 2023](https://arxiv.org/abs/2111.05791), and it comes with the following benefits:

1. Estimation of a transport map $G$ to push base distribution $F_{\boldsymbol U}$ to a target distribution $\tilde F_{\boldsymbol Z}$ (supposely close to $F_{\boldsymbol Z}$) on an independent holdout sample;
2. Personalized inference with rank matching;
3. Distribution-preserving perturbation for sampling diversity and privacy protection.

Moreover, sampling properties of PASS generator make sure that it can be used for simulation-based inference, or Monte Carlo (MC) inference.
This repo focuses on the MC inference examples provided in the paper.

## Experiment

```
git clone https://github.com/yifei-liu-stat/pass.git

```

### Assess Image Generators
Test the generation quality of image generative models on CIFAR-10 with FID score as the test statistic.
MC inference allows us to quantify the uncertainty in FID calculation when inference sample size is limited.

<p align="center">
<img src=reports/figures/compare-images.png />
</p>
(E.g. Generator B has a slightly smaller FID than A does, but the test shows it is not statistically significant)




### Word Importance in Sentiment Analysis

### Multimodal Inference


## BibTex

```
@article{liu2023perturbation,
  title={Perturbation-Assisted Sample Synthesis: A Novel Approach for Uncertainty Quantification},
  author={Liu, Yifei and Shen, Rex and Shen, Xiaotong},
  journal={arXiv preprint arXiv:2305.18671},
  year={2023},
  url={https://arxiv.org/abs/2305.18671}
}
```