## [AllState Claims Severity Challenge](https://www.kaggle.com/c/allstate-claims-severity/)

### Requirements:
* [PyTorch](http://www.pytorch.org)
* [NumPy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* GPU (OPTIONAL, yet recommended)

### Default hyper-parameters (similar to the paper):
* Per-GPU `batch_size` = 1024
* Initial `learning_rate` = 0.02
* Step `lr_decay` gamma = 0.5

### Create the following directory structure:
```
data/
    checkpoints/
    subm/
    ...
```

### Run the training:
Run `python train.py --help` for more info.
