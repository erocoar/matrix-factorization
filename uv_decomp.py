import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
       
np.random.seed(281123)

class UMdecomp:
    def __init__(self):
        self.K = 10            # no of features
        self.num_iter = 75
        self._lambda = 0.05    # regularization parameter
        self.eta = 0.005       # learning rate
        
        self.errs = []
        
    def train(self, X, rm, cm):
        self.global_mean = X.rating.mean()
        self.X = X.pivot(index="userID", columns="movieID", values="rating")
        X = self.X.values
        X_nan = np.where(~np.isnan(X))
        I = X.shape[0]
        J = X.shape[1]
        K = self.K
        
#        U = (np.ones((I, K)) * np.sqrt(self.global_mean / K)) + np.random.normal(size=(I, K)) # commented out: for uncentered data
#        M = (np.ones((K, J)) * np.sqrt(self.global_mean / K)) + np.random.normal(size=(K, J)) # 
        U = np.zeros((I, K)) + np.random.normal(size=(I, K)) # for double centered data
        M = np.zeros((K, J)) + np.random.normal(size=(K, J)) # ^
        
        y = X[X_nan]
        
        for _ in range(self.num_iter):
            for i, j, _iter in zip(*X_nan, np.arange(X_nan[0].size)):
                x_ij = X[i, j]
                u_ik = U[i, :]
                m_kj = M[:, j]
                e_ij = x_ij - u_ik @ m_kj
                
                grad_u = u_ik + self.eta * (2 * e_ij * m_kj - self._lambda * u_ik)
                grad_m = m_kj + self.eta * (2 * e_ij * u_ik - self._lambda * m_kj)
                
                U[i, :] = grad_u
                M[:, j] = grad_m
                
                if _iter % 10000 == 0:
                    err = np.sqrt(np.square((y - (U @ M)[X_nan])).mean())
                    self.errs.append(err)
                    print(str(_) + "/" + str(self.num_iter) + "||", np.around(err, decimals = 2),
                          str(round(_iter / X_nan[0].size, 4) * 100) + "%")
                
        self.U = U
        self.M = M
        self.X.loc[:, :] = self.U @ self.M
        self.X = self.X.values
        
        cm[np.isnan(cm)] = np.nanmean(cm) # comment out for without double centering 
        rm[np.isnan(rm)] = np.nanmean(rm) # comment out for without double centering 
        self.X += cm # comment out for without double centering 
        self.X += rm[:, None] # comment out for without double centering 
        
    def predict(self, X): # this is not ideal and relies on the pivots being equal always to avoid long loops
        X = X.pivot(index="userID", columns="movieID", values="rating").values
        X_nan = np.where(~np.isnan(X))        
        return self.X[X_nan]
          

kf = KFold(n_splits=5, shuffle=True, random_state = 15523)
rmse_train_list = []
mae_train_list  = []
rmse_test_list = []
mae_test_list  = []

run_errars = []
    
for train_idx, test_idx in kf.split(ratings):
    train = ratings.copy()
    test  = ratings.copy()
    
    train = train.sort_values(["userID", "movieID"])
    test  = test.sort_values(["userID", "movieID"])
       
    train.iloc[test_idx, 2] = np.nan
    test.iloc[train_idx, 2] = np.nan
    
    train_piv = train.pivot(index="userID", columns="movieID", values="rating").values
    test_piv  = test.pivot(index="userID", columns="movieID", values="rating").values
    
    train_na  = np.where(~np.isnan(train_piv))
    test_na   = np.where(~np.isnan(test_piv))
    
    rowmeans  = np.nanmean(train_piv, axis=1)
    train_piv = train_piv - rowmeans[:, None]
    
    colmeans  = np.nanmean(train_piv, axis=0)
    train_piv = train_piv - colmeans
    
    train_centered = train.copy() # comment out for without double centering 
    train_centered.rating[train_centered.rating.notnull()] = train_piv[train_na] # comment out for without double centering 
    
    m = UMdecomp()
    m.train(train_centered, rowmeans, colmeans) # change to train, rowmeans, colmeans for without double centering
    pred = m.predict(test)
    
    pred += (test.rating.mean() - pred.mean()) # comment out for without mean shift 
    
    rmse_test  = np.sqrt(np.square(test_piv[test_na] - pred).mean())
    mae_test   = np.abs(test_piv[test_na] - pred).mean()
    
    pred = m.predict(train)
    
    pred += (train.rating.mean() - pred.mean())  # comment out for without mean shift 
    
    rmse_train = np.sqrt(np.square(train.rating[train.rating.notnull()].values - pred).mean())
    mae_train  = np.abs(train.rating[train.rating.notnull()].values - pred).mean()
    
    print("RMSE_TRAIN ==================== " + str(rmse_train))
    
    rmse_test_list.append(rmse_test)
    mae_test_list.append(mae_test)
    rmse_train_list.append(rmse_train)
    mae_train_list.append(mae_train)
    
    run_errars.append(m.errs)
    
np.vstack([rmse_train_list, rmse_test_list, mae_train_list, mae_test_list]).mean(axis=1)