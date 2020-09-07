from shutil import copyfile

import pandas as pd

# Filtering original Kaggle dataset to 15 artist used in original article.
# Result of this script will create new set with selected artist which will be further
# processed.

dest_set_path = '../../dataset/our_set/'

def filter_artists(df):
    artists_group = ["Albrecht Durer",
                     "Boris Kustodiev",
                     "Camille Corot",
                     "Camille Pissarro",
                     "Childe Hassam",
                     "Claude Monet",
                     "Edgar Degas",
                     "Eugene Boudin",
                     "Giovanni Battista Piranesi",
                     "Gustave Dore",
                     "Henri Matisse",
                     "Ilya Repin",
                     "Ivan Aivazovsky",
                     "Ivan Shishkin",
                     "John Singer Sargent"]
    df = df[df.artist.isin(artists_group)]
    return df

df = pd.read_csv('../../all_data_info.csv')
df = filter_artists(df)

# Export our data to csv
df.to_csv("../../our_data.csv")

num_of_pics = 0
for row in df.iterrows():
    path = row[1]['new_filename']
    dir = "train/" if row[1]['in_train'] else "test/"
    copyfile(dir+path, dest_set_path+path)

print("Number of pics: " + str(num_of_pics))



