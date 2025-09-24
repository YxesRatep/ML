# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %% [markdown]
# # Task 1: Train a Naive Bayes Classifier
# The class `nb_train` is located in `a01_functions.py`. You can make
# experimental changes to that class in the other file (`a01_functions.py`). All
# saved changes will be automatically reflected here due to the IPython
# autoreload extension (see below).

# %%
import math

import numpy as np
import numpy.random
# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import nb_train

# %% [markdown]
# # Load the data
# ## Inspect the data

# %%
# Example: show first digit
nextplot()
showdigit(X[0,])
print(y[0])

# %%
# Example: show 15 random digits per class from training data
nextplot()
showdigits(X, y)

# %%
# Example: show a specific set of digits
nextplot()
showdigits(X[0:50,], y[0:50])

# %%
# A simple example dataset that you can use for testing
Xex = np.array([1, 0, # -> 0
                0, 1, # -> 1
                1, 1, # -> 2
                2, 0] # -> 0
              ).reshape(4, 2)
yex = np.array([0, 1, 2, 0])

# %% [markdown]
# # 1 Training

# %%
# Test your code (there should be a warning when you run this)
model = nb_train(Xex, yex, alpha=1)
model
# This should produce:
# {'logcls': array([[[       -inf, -0.69314718, -0.69314718],
#          [ 0.        ,        -inf,        -inf]],
#
#         [[ 0.        ,        -inf,        -inf],
#          [       -inf,  0.        ,        -inf]],
#
#         [[       -inf,  0.        ,        -inf],
#          [       -inf,  0.        ,        -inf]]]),
#  'logpriors': array([-0.69314718, -1.38629436, -1.38629436])}

# %%
import timeit

# Test your code (this time no warning)
model = nb_train(Xex, yex, alpha=2)  # here we use add-one smoothing
print(model)
# This should produce:
# {'logcls': array([[[-1.60943791, -0.91629073, -0.91629073],
#          [-0.51082562, -1.60943791, -1.60943791]],
#
#         [[-0.69314718, -1.38629436, -1.38629436],
#          [-1.38629436, -0.69314718, -1.38629436]],
#
#         [[-1.38629436, -0.69314718, -1.38629436],
#          [-1.38629436, -0.69314718, -1.38629436]]]),
#  'logpriors': array([-0.84729786, -1.25276297, -1.25276297])}

def test_train_perf():
    nb_train(Xex, yex, alpha=2)

runtime = timeit.timeit(test_train_perf, number=100)
print(f"Runtime: {runtime/100:6f}s")

# %%
