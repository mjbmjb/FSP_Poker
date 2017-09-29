#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:52:24 2017

@author: mjb
"""

class Node:
    def __init__(self, data):
        self.child = Node
        self.data = data        
a = Node(1)
b = Node(2)
a.child = b
