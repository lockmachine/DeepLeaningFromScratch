#!/usr/bin/env python3
# coding: utf-8
import numpy as np

class SGD:
    def __init__(self, lr):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
