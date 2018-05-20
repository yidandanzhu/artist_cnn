import os
import random
from collections import defaultdict

classes = ['Picasso', 'vangogh']
num_classes = len(classes)

image_folder = 'artist_dataset'
train_folder = 'artist_dataset_train'
test_folder  = 'artist_dataset_test' 
test_ratio = 0.2

fnames = []
for c in classes:
    path = image_folder + '/' + c
    fs = [(f.replace(' ', '\ '), c) for f in os.listdir(path) if f[0]!='.']
    fnames += fs
    
random.shuffle(fnames)

n_test       = int(len(fnames)*test_ratio)
print('test total:{}'.format(n_test))
fnames_train = fnames[n_test:]
fnames_test  = fnames[:n_test]

# Copy file to folders
os.system('rm -rf {} {}'.format(train_folder, test_folder))
os.system('mkdir {} {}'.format(train_folder, test_folder))
for c in classes:
    os.system('mkdir {}/{} {}/{}'.format(train_folder, c, test_folder, c))
    
n_train = defaultdict(int)
for f, c in fnames_train:
    n_train[c] += 1
    cmd = 'cp {}/{}/{} {}/{}/{}'.format(image_folder, c, f, train_folder, c, f)
    os.system(cmd)
print('train: {}'.format(n_train))

n_test = defaultdict(int)
for f, c in fnames_test:
    n_test[c] += 1
    cmd = 'cp {}/{}/{} {}/{}/{}'.format(image_folder, c, f, test_folder, c, f)
    os.system(cmd)
print('test: {}'.format(n_test))