# # Creating Train / Val / Test folders (One time use)
import os
import shutil

import numpy as np
import pandas as pd

# Script for creating data set for CNN training. NOTE: Delete new_dir_prefix folder after as its
# not used after.

val_ratio = 0.1
test_ratio = 0.1

root_dir = '../dataset/our_cnn_set/'
base_dir ='../../dataset/cnn_dataset/'
dir_prefix = '../../dataset/our_set'
new_dir_prefix = '../../dataset/our_cnn_set/'

def intermediate_set_step():
    df = pd.read_csv('../../our_data.csv')

    num_of_pics = 0
    for row in df.iterrows():
        path = '{dir_prefix}{file_name}'.format(file_name=row[1]['new_filename'], dir_prefix=dir_prefix)
        shutil.copyfile(path, '{new_dir_prefix}{artist}/{file_name}'.format(artist=row[1]['artist'], file_name=row[1]['new_filename'], new_dir_prefix=new_dir_prefix))
        num_of_pics += 1

    print("Number of pics: " + str(num_of_pics))

def split(input_dir):
    # Creating partitions of the data after shuffeling
    allFileNames = os.listdir(base_dir+input_dir)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [name for name in train_FileNames.tolist()]
    val_FileNames = [name for name in val_FileNames.tolist()]
    test_FileNames = [name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        dest_fpath = '{base_dir}{set}/{input_dir}/{name}'.format(base_dir=base_dir, set='train', input_dir=input_dir, name=name)
        src_fpath = '{root_dir}+{input_dir}/{name}'.format(root_dir=root_dir, input_dir=input_dir, name=name)
        os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
        shutil.copy(src_fpath, dest_fpath)

    for name in val_FileNames:
        dest_fpath = '{base_dir}{set}/{input_dir}/{name}'.format(base_dir=base_dir, set='validation', input_dir=input_dir,
                                                                 name=name)
        src_fpath = '{root_dir}+{input_dir}/{name}'.format(root_dir=root_dir, input_dir=input_dir, name=name)
        os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
        shutil.copy(src_fpath, dest_fpath)


    for name in test_FileNames:
        dest_fpath = '{base_dir}{set}/{input_dir}/{name}'.format(base_dir=base_dir, set='test', input_dir=input_dir,
                                                                 name=name)
        src_fpath = '{root_dir}+{input_dir}/{name}'.format(root_dir=root_dir, input_dir=input_dir, name=name)
        os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
        shutil.copy(src_fpath, dest_fpath)


intermediate_set_step()
allFileNames = os.listdir(new_dir_prefix)
for name in allFileNames:
    split(name)
