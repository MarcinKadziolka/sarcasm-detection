# Sarcasm Detection Using Transformer-Based Methods: A Review
## Overview

Sarcasm detection is a challenging task in natural language processing (NLP), with applications in online communication, sentiment analysis, and virtual assistants. 
In this project, we focus on detecting sarcasm in news headlines using transformer-based models. 
We compare our results with a baseline architecture and a state-of-the-art (SOTA) model. 
Our goal is to achieve comparable performance using various transformer architectures. 
We experiment with different attention mechanisms, including default attention, wide attention, and a fast attention mechanism that reduces computational complexity. 
Additionally, we explore the impact of word embeddings on model performance. 
We evaluate our models on a dataset of sarcastic and non-sarcastic news headlines and use binary cross entropy as the loss criterion.

### Introduction

Text classification is a classical task in NLP and machine learning where the objective is to assign predefined categories or labels to a text based on its content. 
Sarcasm detection is a specific type of text classification problem where the goal is to predict whether the text is sarcastic. 
Sarcasm, often characterized by saying something contrary to the intended meaning, is challenging to detect in written text even for humans.

This project utilizes the News Headlines Dataset introduced by Misra and Arora (2023). 
The dataset consists of headlines from two news websites: The Onion and HuffPost, with The Onion providing sarcastic articles about real events.

### Related Work

Various methodologies have been employed for sarcasm detection in textual data across different datasets. 
Prior works have used convolutional neural networks (CNNs), long short-term memory (LSTM) networks, 
bidirectional encoder representations from transformers (BERT), and graph convolutional networks (GCN) for sarcasm detection. 
We use these as a baseline for comparison.
### Model

Our model is a transformer-based architecture. We use token and positional embeddings, apply attention mechanisms, and use a max pooling layer over the time dimension. The output is then compressed by a linear layer and passed through a sigmoid function to classify the headline as sarcastic or not.

### Attention Mechanisms

- Default Attention: Multi-head attention as described in Vaswani et al. (2017). Each head computes attention values, which are then concatenated and linearly transformed.
- Wide Attention: Uses larger dimension sizes for keys and values, potentially capturing more complex information.
- Fast Attention: Implements a simplified attention mechanism from Wu et al. (2021) to reduce computational complexity.

### Word Embeddings

We learn embeddings for each word from scratch. Different embedding dimensions are tested to find the optimal value.

### Positional Embeddings

Since the sequence is important in language, positional embeddings are added to word embeddings to provide the model with information about the order of words.

## Experimental Setup
### Dataset

The dataset is split into training (80%), validation (10%), and test (10%) sets. The training set contains 21,366 examples, with balanced labels for sarcastic and non-sarcastic headlines.

### Hyperparameters

- Batch size: 256
- Learning rate: 0.001
- Weight decay: 0.0001
- Loss criterion: Binary cross entropy
- Optimizers tested: Adam, AdamW, Adadelta, SGD

### Fine-tuning Pre-trained Models

We fine-tuned several pre-trained transformer-based models including GPT-2, BERT, and DistilBERT. These models were trained on the sarcasm detection task using the same dataset.

### Results

We conducted several experiments testing different attention types, depths, and sizes in the transformer architecture. 
Our best performing vanilla transformer model achieved 85% test accuracy. 
Pre-trained models, especially BERT with a classification head, significantly outperformed vanilla transformers with a highest accuracy of 94%.

#### Attention Comparison

| Attention | Number of heads | Depth | Embedding size | Test accuracy |
| --------- | --------------- | ----- | -------------- | ------------- |
| Default   | 10              | 10    | 400            | **85.0%**     |
| Default   | 6               | 6     | 384            | 84.3%         |
| Default   | 6               | 8     | 384            | 84.2%         |
| Default   | 1               | 6     | 384            | 83.9%         |
| Wide      | 6               | 6     | 384            | 83.7%         |
| Default   | 6               | 1     | 384            | 83.0%         |
| Default   | 1               | 1     | 384            | 82.9%         |
| Fast      | 1               | 6     | 384            | 80.3%         |
| Wide      | 1               | 6     | 384            | 80.1%         |
| Fast      | 6               | 6     | 384            | 80.1%         |


#### Final Comparison Table
| Model                                     | Test Accuracy |
| ----------------------------------------- | ------------- |
| Hybrid NN (Misra & Arora, 2023)           | 89.7%         |
| BERT-GCN (Mohan et al., 2023)             | 90.7%         |
| Fine-tuned GPT-2                          | 85.3%         |
| Fine-tuned BERT + Classification Head     | 94%           |
| Fine-tuned Frozen BERT + Classification Head | 67%       |
| Fine-tuned DistilBERT + Classification Head | 85%       |
| Best Vanilla Transformer                  | 85%           |


## Conclusions

Transformer-based architectures are effective for sarcasm detection. Simple architectures perform well despite their low complexity, while fast attention mechanisms, though computationally efficient, result in lower accuracy. Pre-trained models significantly outperform vanilla transformers. Further research could involve testing other attention types and pre-trained word embeddings.

## References

- Amir, S. et al. (2016). [Modelling Context with User Embeddings for Sarcasm Detection in Social Media](https://arxiv.org/abs/1607.00976). arXiv:1607.00976
- Misra, R., & Arora, P. (2023). [Sarcasm detection using news headlines dataset](https://www.sciencedirect.com/science/article/pii/S2666651023000013).
- Vaswani, A. et al. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762). arXiv:1706.03762
- Devlin, J. et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv:1810.04805
- Sanh, V. et al. (2019). [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). arXiv:1910.01108
- Wu, C. et al. (2021). [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084). arXiv:2108.09084
- Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe).
- Kingma, D. P., & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). arXiv:1412.6980
- Loshchilov, I., & Hutter, F. (2017). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). arXiv:1711.05101
- Zeiler, M. D. (2012). [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701). arXiv:1212.5701
