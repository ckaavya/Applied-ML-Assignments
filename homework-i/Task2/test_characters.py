#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:02:00 2017

@author: amla_srivastava
"""

import io

def test_characters():
    with io.open('input.txt', 'r', encoding ='utf-8') as f:
        text = f.read().rstrip()
    assert len(text) == 6
    
    
with io.open('input.txt', 'r', encoding ='utf-8') as f:
        text = f.read().rstrip()
        



    
