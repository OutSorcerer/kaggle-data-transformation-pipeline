# What is this about?

It is a structured and extensible implementation of a typical data transformation workflow made with [doit build-tool](http://pydoit.org/) illustrated with [Statoil/C-CORE Iceberg Classifier Challenge on Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge).

Assume that your have the following workflow:

1. Download archives with datasets from Kaggle website.

2. Unzip them.

3. Transform data by converting from json to numpy format and adding a third layer to image computed as mean between given two layers (HH and HV bands).

4. Train your model (CNN) and save its coefficients to file.

Imagine that you do not want to push these huge files to your version control system and/or Docker registry. You computer may have much slower internet connection than AWS instances where are training your models so downloading and transforming data each time is much faster for you. 

Imagine that you also want to cache intermediate results of these steps so that re-running pipeline does not require repeating unnecessary actions.

But when something is actually changed (your code or input data) cached data should be invalidated and recomputed.

The straitforward solution is just manually implement these steps and caching logic. The obvious disadvantages of this approach are many conditional statements and complexity of detection that inputs of some step are changed and its requires recomputation.

Such problem is historically adressed by software like [Make build automation tool](https://en.wikipedia.org/wiki/Make_(software)). This is already much better but it is based on shell scripts while [doit task management and automation tool](http://pydoit.org/) also supports Python code in tasks and is easily extended with Python code.

# How doit implementation looks like?

Assuming that [all dependencies are installed](#dependencies) and your [specified your Kaggle credentials](#kaggle-credentials) just run the following shell command:

```shell
doit
```

It looks for `dodo.py` and for configuration there, finds `'default_tasks': ['train']` and launches a task called `train`. Its defenition looks like:

```python
def task_train():
    return {
        'actions': [baseline_model.train],
        'file_dep': ['baseline_model.py', 'data/train.npy']
    }
```

`baseline_model.train` is a Python function that reads data from `data/train.npy` and trains a CNN on them. 

Note that it depends not only on data but on source file (`baseline_model.py`) which is correct because if both data and training code is not changed retraining is not required, but if at least one of the inputs changed the model should be also retrained.

`data/train.npy` in turn is configured as a `target` of a task called `convert_train_to_numpy`:

```python
def task_convert_train_to_numpy():
    return {
        'actions': [baseline_model.convert_train_to_numpy],
        'file_dep': ['baseline_model.py', 'data/train.json'],
        'targets': ['data/train.npy']
    }
```

By combining target and dependency files doit tool is able to determine which task depends on which and which task target is up-to-date. This is a default workflow that could be easily extended by implementing a custom Python function to check if task is outdated or not.

# Dependencies

I used [Anaconda](https://anaconda.org/anaconda/python) distribution that already includes packages like [numpy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas) and [scikit-learn](https://github.com/scikit-learn/scikit-learn).

This sample is also using [Keras](https://github.com/fchollet/keras), [TensorFlow](https://github.com/tensorflow/tensorflow), [doit](https://github.com/pydoit/doit), [progressbar](https://github.com/WoLpH/python-progressbar) and [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup) that are all easily installed by `pip`/`conda` except if you want to install [Tensorflow with GPU support](https://www.tensorflow.org/install/) and especially [on Windows](https://www.tensorflow.org/install/install_windows).

# Kaggle credentials

Kaggle requires your login and password to download a dataset. Specify your credentials with environment variables `KAGGLE_LOGIN` and `KAGGLE_PASSWORD` or `.credentials.ini` in the following format (file has higher priority):

```ini
[kaggle]
login=your_login
password=your_password
```

Be carefult and never push this file to a public repository like GitHub (it is already added to `.gitignore` for convinience).

# Credits

Thanks to Gerrit Gruben ([@uberwach](https://github.com/uberwach)) for his idea of using makefiles for data transformation on Kaggle competiontions. See his [Kaggle competition project structure template](https://github.com/uberwach/cookiecutter-kaggle).

Thanks to [DeveshMaheshwari](https://www.kaggle.com/devm2024) for his [kernel with basic CNN model](https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d/notebook).

Thanks to [Kaggle-CLI](https://github.com/floydwch/kaggle-cli) contributors.
