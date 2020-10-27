# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:14:32 2020

@author: sazgar
"""
import gdown

import os.path
if(os.path.isfile('model/datset.csv')):
    print('model exists')

else:
    print('model not found')
    url = 'https://drive.google.com/file/d/1PhSwy7YyUK3_pbwwp-6oUKcvJ6jAEbTU/view?usp=sharing'
    output = 'model/datset.csv'
    gdown.download(url, output, quiet=False) 
    print('model downloaded')