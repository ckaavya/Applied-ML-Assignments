#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:26:16 2017

@author: amla_srivastava
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
    
def test_cv():
    
    
    iris = load_iris()
    X,y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y)
    param_grid = {'n_neighbors' : np.arange(1,15,2)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid = param_grid, cv = 5)
    grid.fit(X_train, y_train)
    
    score = grid.best_score_
    
    assert score >= 0.70