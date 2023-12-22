# MCDropout_MPI

## Context
This code is based on the work carried out in the paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2016) and developed in [Yarin Gal's github repository](https://github.com/yaringal/DropoutUncertaintyExps). They showed that dropout in multilayer perceptron models can be interpreted as a Bayesian approximation. They are able to approximate the epistemic uncertainty in deep learning without a trade-off on performance.

I have applied this approach to address a well-explored regression problem in the aerospace domain [AIAA Scitech Forum 2024](Scitech link). If the obtained results are satisfying, the computational time required to make predictions with such models are relatively heavy. In this repository, I use MPI procedures to speed-up the prediction task on a [Kaggle example](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms). The emphasis is not placed on training the model or enhancing accuracy; instead, the primary objective is to accelerate the prediction phase. Consequently, a standard architecture with default values has been adopted.

## Requirements
The experiment uses Tensorflow 2.10.1, Keras 2.10.0 and mpi4py 3.1.4 (find additional installation info [here](https://mpi4py.readthedocs.io/en/latest/install.html)). It also depends on numpy, pandas, matplotlib, scikit-learn and time libraries.

## BNN with Monte-Carlo dropout

Quick explanation and equations (uses Scitech paper but reformulate). Show CV i.e. need for large amount of samples, and say it's the limiting factor.

The method presented in [Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142) for training Bayesian Neural Networks (BNNs) is known as Monte Carlo Dropout. This technique involves dropout regularisation during training, where a random subset of neurons is temporarily deactivated during each forward pass in an epoch. This creates an ensemble of models, each with a distinct subset of active neurons. During testing, instead of a single forward pass through the network, multiple passes are performed with different subsets of neurons dropped out. The outputs from these passes are averaged to obtain a model prediction, and they are also used to compute the model's predictive uncertainty. Assuming a standard neural network with weights and biases denoted by $\boldsymbol{\theta}$, the network's output for a given input $\mathbf{x}$ is represented as $\hat{y}(\mathbf{x};\boldsymbol{\theta})$. In the BNN model described here, the $T$ forward passes with different dropout masks yield a set of outputs ${\hat{y}(\mathbf{x};\boldsymbol{\theta}t)}{t=1}^T$. The predictive distribution over the network's outputs is approximated by averaging this set of outputs,
$$
    p(y^*| \mathbf{x^*}, \mathbf{x}, y) \approx \frac{1}{T}\sum_{t=1}^{T} p(y^*| \mathbf{x^*}, \boldsymbol{\theta}_t)
$$
The mean and variance of the predictive distribution can be approximated by,
$$
    \mu_{\text{MCD}}(\mathbf{x}) \approx \frac{1}{T}\sum_{t=1}^{T} \hat{y}(\mathbf{x};\boldsymbol{\theta}_t)
$$
$$
    \sigma_{\text{MCD}}^2(\mathbf{x}) \approx \frac{1}{T-1}\sum_{t=1}^{T} \bigl(\hat{y}(\mathbf{x};\boldsymbol{\theta}_t)-\mu_{\text{MCD}}\bigl)^2
$$

The two figures below illustrate the convergence of $\mu_{\text{MCD}}$ and $\sigma_{\text{MCD}}$ predicted by this BNN as the number of Monte Carlo samples increases. Both values are computed based on the validation set. Notably, the convergence is evident, with both parameters stabilizing within a 1$%$ range of $\mu_{\text{MCD}}$ after approximately X samples.

## MPI

Explain briefly what is done and show the results.
