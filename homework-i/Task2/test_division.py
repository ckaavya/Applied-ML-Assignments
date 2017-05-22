#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:54:31 2017

@author: amla_srivastava
"""
from __future__ import division
import numpy as np

def test_division():
    assert 2/8 == 0.25
    
    dividend = np.array([2])
    divisor =  np.array([8])
    
    assert np.true_divide(dividend, divisor) == 0.25
    
