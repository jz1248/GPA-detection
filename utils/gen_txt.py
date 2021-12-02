#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/7/7

@author: Jonathan Zhang 
"""
import os
from glob import glob

root_dir = './leftImg8bit'

for split_name in ['train', 'test', 'val']:
    split_root = os.path.join(root_dir, split_name)
    file_list = glob(split_root + '/*/*.png')

    with open(f"{split_name}.txt", 'w') as f:
        for file in file_list:
            f.write(file[len(split_root)+1:-4] + '\n')
        # f.writelines(file_list)

