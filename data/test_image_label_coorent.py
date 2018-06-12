<<<<<<< HEAD
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/image_2'
label_dir = './ROE/training/label_2'

img_idx = []
for filename in os.listdir(image_dir):
	img_idx.append(filename[:5])

lbl_idx = []
for filename in os.listdir(label_dir):
	lbl_idx.append(filename[:5])

if len(img_idx) != len(lbl_idx):
	if len(img_idx) > len(lbl_idx):
		upper = img_idx
		lower = lbl_idx
	else:
		upper = lbl_idx
		lower = img_idx

	for i in range(len(upper)-1):
		if upper[i] != lower[i]:
			print('error at {}.', format(upper[i]))
			break
else:
	print('Coherent content')
=======
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/image_2'
label_dir = './ROE/training/label_2'

img_idx = []
for filename in os.listdir(image_dir):
	img_idx.append(filename[:5])

lbl_idx = []
for filename in os.listdir(label_dir):
	lbl_idx.append(filename[:5])

if len(img_idx) != len(lbl_idx):
	if len(img_idx) > len(lbl_idx):
		upper = img_idx
		lower = lbl_idx
	else:
		upper = lbl_idx
		lower = img_idx

	for i in range(len(upper)-1):
		if upper[i] != lower[i]:
			print('error at {}.', format(upper[i]))
			break
else:
	print('Coherent content')
>>>>>>> master
