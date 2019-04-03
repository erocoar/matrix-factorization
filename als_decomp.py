import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

np.random.seed(274)

class ALS:
    def __init__(self, K):
        self.K = K            # no of features
        self.num_iter = 75
        self._lambda = 0.05    # regularization parameter
                
    def train(self, X):
        self.global_mean = X.rating.mean()
        self.X = X.pivot(index="userID", columns="movieID", values="rating")
        X = self.X
        X_nan = np.where(~np.isnan(X))
        I = X.shape[0]
        J = X.shape[1]
        K = self.K
        
        U = (np.ones((I, K)) * np.sqrt(self.global_mean / K)) + np.random.normal(size=(I, K))
        M = (np.ones((K, J)) * np.sqrt(self.global_mean / K)) + np.random.normal(size=(K, J))
        
        y = X.values[X_nan]
        
        nr_of_ratings_user  = self.X.notnull().sum(axis=1).values
        nr_of_ratings_movie = self.X.notnull().sum(axis=0).values
        
        bool_rating = X.notnull().values
        X = X.values
        
        nshape = X.shape
                
        for _ in range(self.num_iter):
            
            for _u in range(nshape[0]):
                R_i = X[_u, bool_rating[_u, :]]
                M_i = M[:, bool_rating[_u, :]]
                A_i = M_i @ M_i.T + self._lambda * nr_of_ratings_user[_u] * np.eye(K)
                V_i = M_i @ R_i.T
                
                self.asd = A_i
                u_upd = np.linalg.pinv(A_i) @ V_i
                
                U[_u, :] = u_upd
                
            for _m in range(nshape[1]):
                R_j = X[bool_rating[:, _m], _m]
                U_j = U[bool_rating[:, _m], :]
                A_j = U_j.T @ U_j + self._lambda * nr_of_ratings_movie[_m] * np.eye(K)
                V_j = R_j @ U_j
                
                m_upd = np.linalg.pinv(A_j) @ V_j
                
                M[:, _m] = m_upd
                                
            
            err = np.sqrt(np.square((y - (U @ M)[X_nan])).mean())
            print(str(_) + "/" + str(self.num_iter) + "||", np.around(err, decimals = 2))
                
        self.U = U
        self.M = M
        self.X = self.U @ self.M
                
    def predict(self, X): # this is not ideal and relies on the pivots being equal always to avoid long loops
        X = X.pivot(index="userID", columns="movieID", values="rating").values
        X_nan = np.where(~np.isnan(X))        
        return self.X[X_nan]
          

kf = KFold(n_splits=5, shuffle=True, random_state = 275)
rmse_train_list = []
mae_train_list  = []
rmse_test_list = []
mae_test_list  = []

all_errs = []
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
            
    m = ALS(10)
    m.train(train)#
    pred = m.predict(test)
    
    pred += (test.rating.mean() - pred.mean())  # comment out for without mean shift 
    
    rmse_test  = np.sqrt(np.square(test_piv[test_na] - pred).mean())
    mae_test   = np.abs(test_piv[test_na] - pred).mean()
    
    pred = m.predict(train)
    pred += (train.rating.mean() - pred.mean())  # comment out for without mean shift 
    
    rmse_train = np.sqrt(np.square(train.rating[train.rating.notnull()].values - pred).mean())
    mae_train  = np.abs(train.rating[train.rating.notnull()].values - pred).mean()
       
    rmse_test_list.append(rmse_test)
    mae_test_list.append(mae_test)
    rmse_train_list.append(rmse_train)
    mae_train_list.append(mae_train)
        
np.vstack([rmse_train_list, rmse_test_list, mae_train_list, mae_test_list]).mean(axis=1)
