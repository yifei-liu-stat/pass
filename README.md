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

(Tested on Ubuntu 18.04, with 4 TITAN RTX GPUs and CUDA Version 10.2.)

Create conda environment and install dependencies using the following commands:
```bash
git clone https://github.com/yifei-liu-stat/pass.git
cd pass

conda env create -f environment.yml
conda activate pass
poetry install --no-root
```

### Assess Image Generators
Test the generation quality of image generative models on CIFAR-10 with FID score as the test statistic.
MC inference allows us to quantify the uncertainty in FID calculation when inference sample size is limited.

<p align="center">
<img src=reports/figures/compare-images.png />
</p>
(E.g. Generator B has a slightly smaller FID than A does, but the test shows they are not significantly different.)


To perform inference on this example,
1. Run `make data_assess_generators` to download model checkpoints.
    - Checkpoints will be saved to `./assess_generators/ckpt/`.
2. Run `make inf_assess_generators` to perform inference.
    - The inference results (test statistics, P-value) will be printed out.
    - Sample synthetic images will be saved to `./assess_generators/results/fake_images/`.
    - Comparison of null distributions under different inferenc sizes $n = 2050, 5000, 10000$ is visualized in `./assess_generators/results/null_distributions.png`.
    
  


### Word Importance in Sentiment Analysis
Test importance of sentiment (positive/negative/neutral) words in the IMDB review sentiment classification task, utilizing MC inference.
This example features statistical inference with unstructured data (text) and blackbox models (neural networks), which is challenging with traditional statistical inference methods.

<p align="center">
<img src=reports/figures/sentiment_example.png />
</p>
(E.g. How do positive words like "touching" and "wonderful" contribute to the sentiment analysis?)

To perform inference on this example:

1. Run `make data_words_importance` to download data/checkpoints and prepare some artifacts.
    - Datasets, sentiment word list (along with their appearance frequency in the data), sentiment-masked embeddings will be saved in `./words_importance/data/`.
    - Checkpoints will all be saved to `./words_importance/ckpt/`.
2. Run  `make inf_words_importance` to perform inference.
    - The inference results (test statistic, p-value) will be printed out.
    - Visualization of comparison between true and synthetic embeddings is saved to `./words_importance/results/plots/`;
    - Visualization of null distributions under different sentiment maskings is saved to `./words_importance/results/plots/` as well.
    - KS test results against standard Gaussian will be printed out, indicating the difference between the finite-sample distribution and the asymptotical one.


### Multimodal Inference
Compare two sets of images generated by two different prompts based on stable diffusion models.
FID score is used as the test statistic.


<p align="center">
<img src=reports/figures/prompt_images.png />
</p>
(E.g. Prompts 1 and 2 are the same, which is similar to Prompt 3, but very different from Prompt 4. How can we test the difference in terms of visual signals?)

To perform the inference:

1. Run `make data_multimodal` to generate images from prompts for comparison.
    - Generated images will be saved to `./multimodal_inference/data/`.
2. Run `make inf_multimodal` to perform the inference.
    - The inference results (the pair of prompts to be compared, cosine simularity, test statistic and P-value) will be printed out.
    - Synthetic images generated from stable diffusion with different prompts will be displayed in `./multimodal_inference/results/prompt_images.png`.


## Comments
- The MC inference implementation here is a simplified version of PAI presented in the paper, without rank matching and perturbation.
This is equivalent (in terms of inference resutls) to using rank matching and perturbation due to the nice sampling properties of PASS (Lemma 1 of the paper).
- However, for personalized inference, it is crutial to use rank matching for personalized inference, and perturbation for diversity and privacy protection, so relevant utils is provided in this repo as well (WIP).

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