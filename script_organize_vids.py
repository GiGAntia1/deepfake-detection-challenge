#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separate the videos in the original directories to the
'videos' directory by class.

@author: Dale Kube
"""

import pandas as pd
import os
import shutil
import json
from tqdm import tqdm

parent_dir = '/mnt/extHDD/Kaggle/'

# Paths to the training data
# Combine metadata.json files to a single dataframe
meta = pd.DataFrame(columns=['label','split','original','file','filepath','ID'])
for i in range(1,50):
    
    dir_i = parent_dir+'dfdc_train_part_%01d/' % i
    os.chdir(dir_i)
    
    # Add metadata to dataframe
    with open('metadata.json') as f:
        m = json.load(f)
        m = pd.DataFrame.from_dict(m,orient='index')
        m['file'] = m.index
        m['filepath'] = dir_i+m.index
        m['ID'] = m['file'].str.split('.',n=1,expand=True)[0]
        meta = meta.append(m,ignore_index=True)

# Identify videos that do not exist, according to the metadata
meta['file_exists'] = False
for i in tqdm(meta.index):
    meta.loc[i,'file_exists'] = os.path.exists(meta.loc[i,'filepath'])

print(meta.loc[meta['file_exists']==False,'file'])
meta_missing = meta[meta['file_exists']==False]

# Missing from Part 18
# pvohowzowy.mp4
# wipjitfmta.mp4
# wpuxmawbkj.mp4

# Missing from Part 35
# cfxiikrhep.mp4
# dzjjtfwiqc.mp4
# glleqxulnn.mp4
# innmztffzd.mp4
# zzfhqvpsyp.mp4

# Move videos to the corresponding class directory
os.chdir(parent_dir)
missing = []
for i in tqdm(meta.index):
    df = meta.iloc[i]
    old_file_path = df['filepath']
    label = df['label']
    file = df['file']
    new_file_path = './videos/%s/%s' % (label,file)
    try:
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
        shutil.move(old_file_path,new_file_path)
    except FileNotFoundError:
        missing.append([file])
