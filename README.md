**Pytorch CASENet implementation**

This repository includes a PyTorch implementation of *CASENet: Deep Category-Aware Semantic Edge Detection* (https://arxiv.org/abs/1705.09759)

**Usage:**

Download and extract the training dataset from the following link:

https://drive.google.com/open?id=1WAxUz7Dd9_MoYEDonoSyNCPc4wA36OwR

Run ``train.py`` with argument ``-data-root`` pointing to the SBD root directory

To execute this, you must have Python 3.6.*, [PyTorch](http://pytorch.org/), [scikit-image](http://scikit-image.org/), [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) installed. To accomplish this, we recommend installing the [Anaconda](https://www.anaconda.com/download) Python distribution and use conda to install the dependencies, as it follows:

```bash
conda install matplotlib numpy scikit-image
conda install pytorch torchvision cuda80 -c soumith
```