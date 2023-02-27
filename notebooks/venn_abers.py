import pandas as pd
import numpy as np

from typing import List, Tuple, Dict
from dataclasses import dataclass


def nonleftTurn(a, b, c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1, d2) <= 0


def nonrightTurn(a, b, c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1, d2) >= 0


def slope(a, b):
    ax, ay = a
    bx, by = b
    return (by-ay)/(bx-ax)


def notBelow(t, p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    tx, ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)


@dataclass
class VennAbers:
    """
    Venn Abers algorithm
    
    Ref: https://arxiv.org/abs/1511.00213
    """
    
    k: int = None
    
    def process_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Acoording to equation (1) in the paper
        
        P_i = (X_prime, y_cumsum)
        """
        
        sorted_indices = np.argsort(X)
        
        X_new = X[sorted_indices]
        y_new = y[sorted_indices]
        
        X_unique, X_indices, X_counts = np.unique(X_new, return_inverse=True, return_counts=True)
        
        a = np.zeros_like(X_unique)
        np.add.at(a, X_indices, y_new)
        
        w = X_counts
        y_prime = a / w
        y_cumsum = np.cumsum(y_prime * w)
        X_prime = np.cumsum(w)
        self.k = len(X_prime)
        
        return X_prime, y_cumsum, y_prime, X_unique
        
    def compute_s1(self, P: Dict) -> List:
        """
        Algorithm 1 in the paper
        
        """
        
        S = []
        P[-1] = np.array((-1, -1))
        S.append(P[-1])
        S.append(P[0])
        for i in range(1, self.k+1):
            while len(S) > 1 and nonleftTurn(S[-2], S[-1], P[i]):
                S.pop()
            S.append(P[i])
        return S
        
    def compute_f1(self, P: Dict) -> np.ndarray:
        """
        Algorithm 2 in the paper

        """
        
        S = self.compute_s1(P)
        
        S_inv = S[::-1]
        
        # print(S_inv, P)
        
        F1 = np.zeros((self.k+1, ))
        for i in range(1, self.k+1):
            F1[i] = slope(S_inv[-1], S_inv[-2])
            P[i-1] = P[i-2] + P[i] - P[i-1]
            if notBelow(P[i-1], S_inv[-1], S_inv[-2]):
                continue
            S_inv.pop()
            while len(S_inv) > 1 and nonleftTurn(P[i-1], S_inv[-1], S_inv[-2]):
                S_inv.pop()
            S_inv.append(P[i-1])
            
        return F1
    
    def compute_s0(self, P: Dict) -> List:
        """
        Algorithm 3 in the paper
        
        """

        S = []
        S.append(P[self.k+1])
        S.append(P[self.k])
        for i in range(self.k-1, 0-1, -1):
            while len(S) > 1 and nonrightTurn(S[-2], S[-1], P[i]):
                S.pop()
            S.append(P[i])
        return S

    def compute_f0(self, P: Dict) -> np.ndarray:
        """
        Algorithm 4 in the paper
        """
        
        S = self.compute_s0(P)
        
        S_inv = S[::-1]
        
        F0 = np.zeros((self.k+1, ))
        for i in range(self.k, 1-1, -1):
            F0[i] = slope(S_inv[-1], S_inv[-2])
            P[i] = P[i-1] + P[i + 1] - P[i]
            if notBelow(P[i], S_inv[-1], S_inv[-2]):
                continue
            S_inv.pop()
            while len(S_inv) > 1 and nonrightTurn(P[i], S_inv[-1], S_inv[-2]):
                S_inv.pop()
            S_inv.append(P[i])
            
        return F0

    def compute_f(self, X_prime: np.ndarray, y_cumsum: np.ndarray) -> Tuple:
        P = {0: np.array((0, 0))}
        P.update({i+1: np.array((X_prime[i], y_cumsum[i])) for i in range(self.k)})
        
        # compute f1
        F1 = self.compute_f1(P)
        
        P = {0: np.array((0, 0))}
        P.update({i+1: np.array((X_prime[i], y_cumsum[i])) for i in range(self.k)})
        P[self.k+1] = P[self.k] + np.array((1.0,0.0))
        
        # compute f0
        F0 = self.compute_f0(P)
        
        return F0, F1
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model
        
        Args:
            X: predicted scores
            y: true labels
        """
        
        X_prime, y_cumsum, y_prime, X_unique = self.process_data(X, y)
        
        self.F0, self.F1 = self.compute_f(X_prime, y_cumsum)
        
        self.X_unique = X_unique
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the calibrated probability
        
        Args:
            X: predicted scores
            
        Returns:
            calibrated probability
        """
        
        pos0 = np.searchsorted(self.X_unique, X, side="left")
        pos1 = np.searchsorted(self.X_unique[:-1], X, side="right") + 1
        P0 = self.F0[pos0]
        P1 = self.F1[pos1]
        prob = P1 / (1 - P0 + P1)
        return prob
    
    
class VennAbersWrapper:
    def __init__(self, model):
        self.base_model = model
        self.calibrator = VennAbers()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        
        self.base_model.fit(X, y)
        prob = self.base_model.predict_proba(X)[:, 1].flatten()
        
        self.calibrator.fit(prob, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        prob = self.base_model.predict_proba(X)[:, 1].flatten()
        prob = self.calibrator.predict_proba(prob)
        return prob
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        prob = self.predict_proba(X)
        return (prob >= 0.5).astype(int)
