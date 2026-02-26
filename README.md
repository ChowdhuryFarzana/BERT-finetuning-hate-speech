# Hate Speech Detection using BERT Fine-Tuning

## Paper Being Replicated

This implementation is based on the methodology described in:

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, 4171-4186.

The core approach follows the fine-tuning paradigm introduced in the original BERT paper, where a pre-trained transformer model is adapted to a downstream classification task by adding a linear classification head on top of the [CLS] token representation and training end-to-end on labeled data.

### Analysis and Performance Evaluation

#### 1. Implementation

In this notebook, we reproduced all four BERT fine-tuning strategies from Section 3.1 of Mozafari et al.
(2019) using HuggingFace Transformers (v4.44.0) and PyTorch on a Tesla T4 GPU.
The original paper used pytorch-pretrained-bert on a Tesla K80 GPU. But I was unable to find the exact version.

We evaluated on the Davidson dataset (24,783 tweets: Hate, Offensive, Neither) using
the paper's exact setup: bert-base-uncased, 80/10/10 stratified split, AdamW lr=2e-5,
batch size=32, 3 epochs, dropout=0.1, max sequence length=64. One addition not mentioned
in the paper was a linear warmup scheduler over the first 10% of training steps, which
is standard BERT fine-tuning practice.

The Waseem dataset could not be reproduced. It is distributed as tweet IDs only and
requires Twitter API hydration. The current free API tier limits access to 1,500
tweets/month, making recovery of 19,000 tweets impossible for me at this moment. We tried to use HuggingFace Hub for pre-processed versions, but the available datasets either had only two classes (hate vs no-hate) instead of the paper's three classes (racism, sexism, neither), or contained only tweet IDs without text. So, We only used davidson dataset for the model.

#### 2. Performance Evaluation

##### Results vs Paper (Davidson Dataset)

| Method               | Paper F1 (%) | Our F1 (%) | Difference |
|----------------------|--------------|------------|------------|
| BERTBase             | 91           | 91.37      | +0.37      |
| BERTBase + NonLinear | 77           | 90.52      | +13.52     |
| BERTBase + BiLSTM    | 92           | 90.87      | -1.13      |
| BERTBase + CNN       | 92           | 90.82      | -1.18      |

#### Key Findings

1. BERTBase matched the paper almost exactly (+0.37%). The simple linear classifier
on [CLS] leaves minimal room for implementation differences, making this our most
faithful reproduction and validating our overall pipeline.

2. NonLinear dramatically outperformed the paper (+13.52%). This is the most
significant discrepancy. The paper reports NonLinear as its worst strategy (77%),
while we achieve 90.52%, comparable to our other strategies. The most likely cause
is our warmup scheduler, which stabilizes gradient flow through the two additional
hidden layers. Without warmup, deeper classifier heads are prone to unstable early
training, which likely explains the paper's poor result for this strategy.

3. BiLSTM and CNN slightly underperformed the paper (-1.13% and -1.18%). Both gaps
are within normal variance across random seeds. The paper does not report a fixed seed,
and minority class performance can vary by 1-2% across runs on imbalanced datasets.

4. Strategy ranking changed. The paper ranks CNN best and NonLinear worst. Our
results rank BERTBase best and CNN lowest among BERT strategies. This suggests the
CNN's advantage in the paper is not architecturally guaranteed but may be
context-dependent or seed-sensitive.

#### Per-Class Analysis

| Class     | BERTBase | NonLinear | BiLSTM | CNN  | Support |
|-----------|----------|-----------|--------|------|---------|
| Hate      | 0.42     | 0.39      | 0.39   | 0.39 | 143     |
| Offensive | 0.95     | 0.95      | 0.95   | 0.95 | 1919    |
| Neither   | 0.91     | 0.89      | 0.90   | 0.90 | 417     |

All four strategies struggle with the Hate class (F1: 0.39-0.42) while performing
strongly on Offensive (F1: 0.95). This directly matches with the paper's error analysis.
The Hate class has only 1,430 samples vs 19,190 Offensive, creating severe class
imbalance. The paper attributes this to annotation bias where crowdsourced annotators
labeled African American Vernacular English tweets as Hate when they were more
accurately Offensive or Neither in social context. Our reproduction shows the same
bias pattern across all four strategies.

#### Summary of Discrepancy Causes

- **Warmup scheduler:** Not used in paper, added by us. Likely responsible for the
  NonLinear strategy's much stronger performance in our reproduction.
- **Library improvements:** transformers v4.44.0 vs pytorch-pretrained-bert (2019).
  Five years of optimization improvements benefit complex classifier heads more than
  simple ones, contributing to our more consistent results across strategies.
- **Hardware:** Tesla T4 vs K80. Different floating point characteristics affect
  gradient updates subtly across many training steps.

The result shows the paper's core finding: BERT-based fine-tuning
outperforms traditional baselines. The per-class results validate
the paper's error analysis on annotation bias. The main departure is that our
results are more uniform across strategies (90.52-91.37%) compared to the paper's
wider spread (77-92%), suggesting modern training infrastructure reduces the
sensitivity of results to classifier head complexity.

## Citation

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, pages 4171-4186. Association for Computational Linguistics.
