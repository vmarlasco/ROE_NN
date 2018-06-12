<<<<<<< HEAD
#!/usr/bin/python

import numpy as np
import cv2
import os


image_dir = './ROE/training/image_2'

for filename in os.listdir(image_dir):

	assert os.path.exists(os.path.join(image_dir, filename)), \
        'File does not exist: {}'.format(filename)

	try:
		im = cv2.imread(os.path.join(image_dir, filename)).astype(np.float32, copy=False)
	except AttributeError:
		print(filename)
		break
=======
#!/usr/bin/python

import numpy as np
import cv2
import os


image_dir = './ROE/training/image_2'

for filename in os.listdir(image_dir):

	assert os.path.exists(os.path.join(image_dir, filename)), \
        'File does not exist: {}'.format(filename)

	try:
		im = cv2.imread(os.path.join(image_dir, filename)).astype(np.float32, copy=False)
	except AttributeError:
		print(filename)
		break
>>>>>>> master
