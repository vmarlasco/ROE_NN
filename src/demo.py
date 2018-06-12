<<<<<<< HEAD
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from config import *
from nets import *
from train import _draw_box

FLAGS = tf.app.flags.FLAGS

base_dir = "F:/MEGA/THESIS/SOFTWARE/NN/ROE_NN"

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', base_dir + '/data/model_checkpoints/squeezeDet/model.ckpt-400',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', base_dir + '/src/data/sample3.jpg',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', base_dir + '/src/data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


def image_demo():
    """Detect image."""
    print("\n################################ IMAGE DETECTOR #############################")
    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = roe_squeezeDet_config()
            mc.BATCH_SIZE = 1
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDet(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            for f in glob.iglob(FLAGS.input_path):

                assert os.path.exists(f), \
                    'File does not exist: {}'.format(f)

                im = cv2.imread(f)
                im = im.astype(np.float32, copy=False)
                im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                input_image = im - mc.BGR_MEANS

                # Detect
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [input_image]})

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    'car': (255, 191, 0),
                    'cyclist': (0, 191, 255),
                    'pedestrian': (255, 0, 191)
                }

                # Draw boxes
                _draw_box(
                    im, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
                     for idx, prob in zip(final_class, final_probs)],
                    cdict=cls2clr,
                )

                file_name = os.path.split(f)[1]
                out_file_name = os.path.join(FLAGS.out_dir, 'out_' + file_name)
                cv2.imwrite(out_file_name, im)
                print('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    if FLAGS.mode == 'image':
        image_demo()
    else:
        video_demo()


if __name__ == '__main__':
    tf.app.run()
=======
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from config import *
from nets import *
from train import _draw_box

FLAGS = tf.app.flags.FLAGS

base_dir = 'F:/DATASETS/squeezeDet'

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', base_dir + '/data/model_checkpoints/squeezeDet/model.ckpt-400',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', base_dir + '/src/data/sample3.jpg',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', base_dir + '/src/data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


def image_demo():
    """Detect image."""
    print("\n################################ IMAGE DETECTOR #############################")
    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = roe_squeezeDet_config()
            mc.BATCH_SIZE = 1
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDet(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            for f in glob.iglob(FLAGS.input_path):

                assert os.path.exists(f), \
                    'File does not exist: {}'.format(f)

                im = cv2.imread(f)
                im = im.astype(np.float32, copy=False)
                im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                input_image = im - mc.BGR_MEANS

                # Detect
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [input_image]})

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    'car': (255, 191, 0),
                    'cyclist': (0, 191, 255),
                    'pedestrian': (255, 0, 191)
                }

                # Draw boxes
                _draw_box(
                    im, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
                     for idx, prob in zip(final_class, final_probs)],
                    cdict=cls2clr,
                )

                file_name = os.path.split(f)[1]
                out_file_name = os.path.join(FLAGS.out_dir, 'out_' + file_name)
                cv2.imwrite(out_file_name, im)
                print('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    if FLAGS.mode == 'image':
        image_demo()
    else:
        video_demo()


if __name__ == '__main__':
    tf.app.run()
>>>>>>> master
