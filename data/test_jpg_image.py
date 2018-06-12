<<<<<<< HEAD
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/image_2'

for filename in os.listdir(image_dir):
	if not filename.find('.jpg'):
		print('Error at {}'.format(filename))
=======
#!/usr/bin/python

import numpy as np
import os

image_dir = './ROE/training/image_2'

for filename in os.listdir(image_dir):
	if not filename.find('.jpg'):
		print('Error at {}'.format(filename))
>>>>>>> master
		break