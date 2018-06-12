<<<<<<< HEAD
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/images'
image_set_dir = './ROE/ImageSets'
trainval_file = image_set_dir+'/trainval.txt'
train_file = image_set_dir+'/train.txt'
val_file = image_set_dir+'/val.txt'

idx = []
for filename in os.listdir(image_dir):
	idx.append(filename[:5])

idx = np.random.permutation(idx)

with open(train_file, 'w') as f:
	for file_index in idx:
		f.write('{}\n'.format(file_index))
f.close()

print('Trainining set is saved to ' + train_file)
=======
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/images'
image_set_dir = './ROE/ImageSets'
trainval_file = image_set_dir+'/trainval.txt'
train_file = image_set_dir+'/train.txt'
val_file = image_set_dir+'/val.txt'

idx = []
for filename in os.listdir(image_dir):
	idx.append(filename[:5])

idx = np.random.permutation(idx)

with open(train_file, 'w') as f:
	for file_index in idx:
		f.write('{}\n'.format(file_index))
f.close()

print('Trainining set is saved to ' + train_file)
>>>>>>> master
