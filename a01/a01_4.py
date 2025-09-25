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
K = 200
Kf = KFold(n_splits=K, shuffle=True)
alpha_grid = []
for i in range(1):
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
        X_test = X[i_test]
        
        y_i = y[i_train]
        y_test = y[i_test]
    
        # print(f"{y_test = }")
        
        model = nb_train(X_i, y_i, alpha=2)
        y_pred = nb_predict(model, X_test)["yhat"]
    
        # print(f"{y_pred = }")

        # THIS WAS SHIT
        # 0-1 loss
        # oi_loss_arr = np.multiply(y_pred, y_test) #gyatt
        # error = sum(oi_loss_arr)/len(oi_loss_arr)

        tf_vec = np.equal(y_pred, y_test)
        all = len(tf_vec)
        [false, right] = np.bincount(tf_vec)
        error=69
        print(f"{false = }")
        print(f"{right = }")
        print(f"{error = }")
