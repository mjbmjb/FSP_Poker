# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:32:00 2017

@author: walk
"""

from socket import *

soc = socket(AF_INET, SOCK_STREAM)
soc.connect(('127.0.0.1', 20001))


soc.send("VERSION:2.0.0\r\n")  


soc_file = soc.makefile()

print(soc_file.readline())

soc.makefile().readline() 
