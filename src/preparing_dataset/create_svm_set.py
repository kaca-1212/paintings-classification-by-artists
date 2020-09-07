from shutil import copyfile

import pandas as pd

# Converting our filtered data set to set which will used for svm training and testing.

dest_set_path = '../../dataset/svm_set/'
df = pd.read_csv('../../our_data.csv')

artists_dict = {"Albrecht Durer":0,
                "Boris Kustodiev":0,
                "Camille Corot":0,
                "Camille Pissarro":0,
                "Childe Hassam":0,
                "Claude Monet":0,
                "Edgar Degas":0,
                "Eugene Boudin":0,
                "Giovanni Battista Piranesi":0,
                "Gustave Dore":0,
                "Henri Matisse":0,
                "Ilya Repin":0,
                "Ivan Aivazovsky":0,
                "Ivan Shishkin":0,
                "John Singer Sargent":0
                }

num_of_pics = 0
for row in df.iterrows():
    path = row[1]['new_filename']
    artist = row[1]['artist']
    dir = "our_set/"
    copyfile(dir+path, dest_set_path + artist + "_" + str(artists_dict[artist]))
    artists_dict[artist] += 1

print("Number of pics: " + str(num_of_pics))
