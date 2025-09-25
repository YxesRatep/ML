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

from tqdm import tqdm

from a01_helper import *
from a01_functions import nb_train, nb_predict, nb_train_fast

# %%
# To create folds, you can use:
K = 20
Kf = KFold(n_splits=K, shuffle=True)

alpha_grid = np.linspace(2,15,60)

for a in tqdm(alpha_grid, desc="Alphas"):
    for i_train, i_test in tqdm(Kf.split(X), desc=f"Folds (alpha={a})", leave=False):
        # code here is executed K times, once per test fold
        # i_train has the row indexes of X to be used for training
        # i_test has the row indexes of X to be used for testing
    
        print(
            "Fold has {:d} training points and {:d} test points".format(
                len(i_train), len(i_test)
            )
        )
        
        X_i = np.take(X, i_train, axis=0)
        y_i = np.take(y, i_train, axis=0)
        X_t = np.take(X, i_test,  axis=0)
        y_t = np.take(y, i_test,  axis=0)
        
        model = nb_train(X_i, y_i, alpha=a)
        y_pred = nb_predict(model, X_t)['yhat']
    
        #pred_vec = np.equal(y_pred,y_t)
        #false, true = np.bincount(pred_vec)
        #tpr = true/len(pred_vec)
        acc = np.mean(y_pred == y_t)
        
        print(acc)

# %%

# %%
# Use cross-validation to find a good value of alpha. Also plot the obtained
# accuracy estimate (estimated from CV, i.e., without touching test data) as a
# function of alpha.
# YOUR CODE HERE
