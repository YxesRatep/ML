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
import numpy as np
from a01_helper import logsumexp


# %%
def nb_train(X, y, alpha=1, K=None, C=None):
    """Train a Naive Bayes model.

    We assume that all features are encoded as integers and have the same domain
    (set of possible values) from 0:(K-1). Similarly, class labels have domain
    0:(C-1).

    Parameters
    ----------
    X : ndarray of shape (N,D)
        Design matrix.
    y : ndarray of shape (N,)
        Class labels.
    alpha : int
        Parameter for symmetric Dirichlet prior (Laplace smoothing) for all
        fitted distributions.
    K : int
        Each feature takes values in [0,K-1]. None means auto-detect.
    C : int
        Each class label takes values in [0,C-1]. None means auto-detect.

    Returns
    -------
    A dictionary with the following keys and values:

    logpriors : ndarray of shape (C,)
        Log prior probabilities of each class such that logpriors[c] contains
        the log prior probability of class c.

    logcls : ndarray of shape(C,D,K)
        A class-by-feature-by-value array of class-conditional log-likelihoods
        such that logcls[c,j,v] contains the conditional log-likelihood of value
        v in feature j given class c.
    """
    N, D = X.shape
    if K is None:
        K = np.max(X) + 1
    if C is None:
        C = np.max(y) + 1

    # Compute class priors and store them in priors
    counts = np.bincount(y, minlength=C).astype(float)
    priors = (counts + (alpha - 1)) / (N + C * (alpha - 1))
    
    # Compute class-conditional densities in a class x feature x value array
    # and store them in cls.
    cls = np.zeros((C, D, K))

    """
    cls[a,b,c] = P(feat_b = c | a)
    """
    
    # HERE
    for c in range(C):
        Xc = X[y == c]
        Nc = Xc.shape[0]
        for j in range(D):
            vcounts = np.bincount(Xc[:, j], minlength=K).astype(float)
            cls[c, j, :] = (vcounts + alpha - 1) / (Nc + K * (alpha - 1))

    return dict(
        logpriors=np.log(priors),
        logcls=np.log(cls),
    )


# %%
import numpy as np

def nb_train_fast(X, y, C=None, K=None, alpha=1.0):
    # X: (N, D) ints in [0, K-1]
    # y: (N,)  ints in [0, C-1]
    N, D = X.shape
    if K is None:
        K = np.max(X) + 1
    if C is None:
        C = np.max(y) + 1

    alpha -= 1

    y_onehot = np.eye(C, dtype=np.float64)[y]      # (N, C)
    x_onehot = np.eye(K, dtype=np.float64)[X]      # (N, D, K)

    # counts[c, d, k] = sum over N of 1[y_n=c]*1[X_{n,d}=k]
    counts = np.einsum('nc,ndk->cdk', y_onehot, x_onehot, optimize=True)  # (C, D, K)

    Nc = y_onehot.sum(axis=0).reshape(C, 1, 1)     # (C,1,1)

    # Laplace/Dirichlet smoothing: (counts + alpha) / (Nc + K*alpha)
    cls = (counts + alpha) / (Nc + K * alpha)

    # Priors
    priors = (Nc.squeeze() / N)
    logpriors = np.log(np.clip(priors, 1e-300, 1.0))

    return dict(
        logpriors=np.log(priors),
        logcls=np.log(cls),
    )



# %%
def nb_predict(model, Xnew):
    """Predict using a Naive Bayes model.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    Xnew : nd_array of shape (Nnew,D)
        New data to predict.

    Returns
    -------
    A dictionary with the following keys and values:

    yhat : nd_array of shape (Nnew,)
        Predicted label for each new data point.

    logprob : nd_array of shape (Nnew,)
        Log-probability of the label predicted for each new data point.
    """
    logpriors = model["logpriors"]
    logcls = model["logcls"]
    Nnew = Xnew.shape[0]
    C, D, K = logcls.shape

    # print(f"{Xnew.shape = }")
    # print(f"{logpriors.shape = }")
    # print(f"{logcls.shape = }")

    
    # Compute the unnormalized log joint probabilities P(Y=c, x_i) of each
    # test point (row i) and each class (column c); store in logjoint
    
    """
      P(X,Y) = P(X|Y)P(Y) = P(Y|X)P(X)

      log P(Y=c, x) = logpriors[c] + sum_j logcls[c, j, x[j]]
      cls[a,b,c] = P(feat_b = c | a)
    """
    logjoint = np.zeros((Nnew, C))
    for i in range(Nnew):
        for c in range(C):
            logjoint[i, c] = logpriors[c]
            for d in range(D):
                logjoint[i, c] += logcls[c, d, Xnew[i,d]]
                
    # print(logjoint.shape)
    # print(logjoint)
    # print(np.exp(logjoint))
    # logsumexponent := lsex 
    
    lsex = logsumexp(logjoint.T)
    # print(f"{lsex = }")
    
    norm_logjoint = logjoint.T - lsex
    # print(f"{norm_logjoint = }")


    # Compute predicted labels (in "yhat") and their log probabilities
    # P(yhat_i | x_i) (in "logprob")
    yhat = np.argmax(norm_logjoint, axis=0)
    logprob = np.max(norm_logjoint, axis=0)

    
    return dict(yhat=yhat, logprob=logprob)


# %%
def nb_generate(model, ygen):
    """Given a Naive Bayes model, generate some data.

    Parameters
    ----------
    model : dict
        A Naive Bayes model trained with nb_train.
    ygen : nd_array of shape (n,)
        Vector of class labels for which to generate data.

    Returns
    -------
    nd_array of shape (n,D)

    Generated data. The i-th row is a sampled data point for the i-th label in
    ygen.
    """
    logcls = model["logcls"]
    n = len(ygen)
    C, D, K = logcls.shape
    Xgen = np.zeros((n, D))
    for i in range(n):
        c = ygen[i]
        # Generate the i-th example of class c, i.e., row Xgen[i,:]. To sample
        # from a categorical distribution with parameter theta (a probability
        # vector), you can use np.random.choice(range(K),p=theta).
        # YOUR CODE HERE
        """
        cls[c,d,k] = P(feat_d = k | c)
        """
        for d in range(D):
            theta = np.exp(logcls[c, d, :])
            theta /= theta.sum() # normalize
            Xgen[i, d] = np.random.choice(range(K), p=theta)
        
    return Xgen

# %%

# %%
