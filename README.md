# Bias models for NER task

Contributors - [Abhishek Pratap Singh](https://github.com/papi656), [Kishore M](https://github.com/kishore7eng)

### train_bias_model.py
- This file train a single token based biased model.
- The weights after training are saved.

### bias_model.py
- This file is used for two task:
    1. To generate the probability distribution predicted by biased model for each token. This will be used for debiasing
    2. To generate predictions in terms of BIO tags for each token. This will be used for categorising the performance over various catergories of mentions