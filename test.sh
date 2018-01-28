#!/bin/bash  
  
basepath=$(cd `dirname $0`; pwd)  
cd $basepath/Player
python six_player.py  
