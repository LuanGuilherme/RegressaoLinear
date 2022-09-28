import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
from collections import Counter
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as skl
from sklearn.metrics import mean_squared_error
import statsmodels.api as sme

# tratamento do dataset
dataset = pd.read_csv(
    "dataset.csv", sep=",", low_memory=False)  # lendo .csv do dataset

print(dataset)