Birmingham Purchase Cards Transactions
==============================

**Description:** With about 14  explanatory variables that describe (almost) all aspects of the corporate  purchase card transactions behavior for the Birmingham City Council, the idea for this paper is to use machine learning to explore and gain deeper knowledge from this data set.

**Where does this data comes from?:** Under the Code of Recommended Practice for Local Authorities on Data Transparency, councils are encouraged to publish all corporate purchase card transactions. 

Birmingham City Council has already publish details of all their relevant expenditure of more than £500 within Payments to Suppliers page, and will continue to do so. However, in the spirit of the Code, the Council started publishing all purchase card transactions, regardless of value, from the April 2014 card statement. 

Data: https://data.birmingham.gov.uk/dataset/purchase-card-transactions

Kernel Ridge:  https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf


**Problem tasks:** 
* (Clustering) Discovering profiles (whether the case) or unusual transactions (anomalies detection)
* (Forecasting) Try to guess future transactional behaviors. 

# Notebooks

1.Birmingham_preprocess.ipynb - Data loading and cleaning

2.Birmingham_feature_engineer.ipynb  - Data explore and trnasformation

3-Birmingham_model.ipynb (Selected Model) - TimeSeries Model.

Modeling conclusions

4-Birmingham_model_unsupervised.ipynb - Modeling using unsupervised approach


**Selected Measure**

The selected evaluation metric is the 

*Time Series: MSE*


**Selected Model**

The selected model is the result of process detailed in *3.Birmingham_model.ipynb*

The model is saved in the models folder:

Birmingham_ts_model.sav

# Project Arquitecture

![alt text](https://github.com/anarua1203/Birmingham-ml-project/tree/main/ana.rua/docs/BirminghamArquitecture.png?raw=true)

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
