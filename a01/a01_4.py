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
K = 10
Kf = KFold(n_splits=K, shuffle=True)

alpha_grid = np.linspace(2,15,28)



for a in tqdm(alpha_grid, desc="Alphas"):
    acc_acc = 0
    
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
        acc_acc += acc

    mean_acc = acc_acc / K
    print(f"{alpha = }: {mean_acc = }")

# %%
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

K = 10
Kf = KFold(n_splits=K, shuffle=True, random_state=42)   # reproducible
alpha_grid = np.linspace(2, 15, 28)

# Precompute splits once so they’re picklable & reusable
splits = list(Kf.split(X))
n_alphas = len(alpha_grid)
n_folds = len(splits)

def run_pair(alpha_idx, a, fold_idx, train_idx, test_idx):
    X_i, y_i = X[train_idx], y[train_idx]
    X_t, y_t = X[test_idx], y[test_idx]
    model = nb_train(X_i, y_i, alpha=a)
    y_pred = nb_predict(model, X_t)['yhat']
    acc = float(np.mean(y_pred == y_t))
    return alpha_idx, fold_idx, acc

tasks = [
    (i, a, j, tr, te)
    for i, a in enumerate(alpha_grid)
    for j, (tr, te) in enumerate(splits)
]

# Threads are the safest default if your code imports IPythony stuff or uses NumPy a lot
# Set BLAS threads to 1 to avoid oversubscription: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(run_pair)(i, a, j, tr, te) for (i, a, j, tr, te) in tqdm(tasks, desc="α×folds")
)

# Aggregate to [n_alphas, n_folds] then mean over folds
acc_mat = np.empty((n_alphas, n_folds), dtype=float)
for i, j, acc in results:
    acc_mat[i, j] = acc

mean_acc = acc_mat.mean(axis=1)
best_idx = int(np.argmax(mean_acc))
best_alpha = float(alpha_grid[best_idx])

for a, m in zip(alpha_grid, mean_acc):
    print(f"alpha={a:.3f}, mean_acc={m:.4f}")

print(f"\nBest alpha = {best_alpha:.3f} with mean_acc = {mean_acc[best_idx]:.4f}")


# %%
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

# --- Setup logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

K = 10
Kf = KFold(n_splits=K, shuffle=True, random_state=42)
alpha_grid = np.linspace(2, 15, 28)

# Precompute splits (picklable + reusable)
splits = list(Kf.split(X))
n_alphas = len(alpha_grid)
n_folds = len(splits)

def run_pair(alpha_idx, a, fold_idx, train_idx, test_idx):
    logger.debug(
        f"Starting task: alpha={a:.3f} (idx={alpha_idx}), "
        f"fold={fold_idx}, train={len(train_idx)}, test={len(test_idx)}"
    )
    X_i, y_i = X[train_idx], y[train_idx]
    X_t, y_t = X[test_idx], y[test_idx]

    model = nb_train(X_i, y_i, alpha=a)
    y_pred = nb_predict(model, X_t)['yhat']
    acc = float(np.mean(y_pred == y_t))

    logger.debug(
        f"Finished task: alpha={a:.3f} (idx={alpha_idx}), "
        f"fold={fold_idx}, acc={acc:.4f}"
    )
    return alpha_idx, fold_idx, acc

# Build all (alpha, fold) jobs
tasks = [
    (i, a, j, tr, te)
    for i, a in enumerate(alpha_grid)
    for j, (tr, te) in enumerate(splits)
]

results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(run_pair)(i, a, j, tr, te) for (i, a, j, tr, te) in tqdm(tasks, desc="α×folds")
)

# Aggregate
acc_mat = np.empty((n_alphas, n_folds), dtype=float)
for i, j, acc in results:
    acc_mat[i, j] = acc

mean_acc = acc_mat.mean(axis=1)
best_idx = int(np.argmax(mean_acc))
best_alpha = float(alpha_grid[best_idx])

for a, m in zip(alpha_grid, mean_acc):
    print(f"alpha={a:.3f}, mean_acc={m:.4f}")

print(f"\nBest alpha = {best_alpha:.3f} with mean_acc = {mean_acc[best_idx]:.4f}")

