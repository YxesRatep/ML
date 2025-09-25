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

# %% [markdown]
# # 4 Model Selection (optional)

# %%
from sklearn.model_selection import KFold
# %load_ext autoreload
# %autoreload 2

from a01_helper import *
from a01_functions import nb_train, nb_predict, nb_train_fast

# %%
# To create folds, you can use:
K = 5
Kf = KFold(n_splits=K, shuffle=True)
for i_train, i_test in Kf.split(X):
    # code here is executed K times, once per test fold
    # i_train has the row indexes of X to be used for training
    # i_test has the row indexes of X to be used for testing

    print(
        "Fold has {:d} training points and {:d} test points".format(
            len(i_train), len(i_test)
        )
    )
    
    X_i = X[i_train]
    X_t = X[i_test]
    y_i = y[i_train]
    y_t = y[i_test]

    model = nb_train_fast(X_i, y_i, alpha=2)
    y_pred = nb_predict(model, X_t)

    # 1-0 loss
    loss = len(np.intersect1d(y_t, y_pred))/len(i_test)
    print(loss)

# %%

# %%
# Use cross-validation to find a good value of alpha. Also plot the obtained
# accuracy estimate (estimated from CV, i.e., without touching test data) as a
# function of alpha.
# YOUR CODE HERE
