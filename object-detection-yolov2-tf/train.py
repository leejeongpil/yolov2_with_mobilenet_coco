import os
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.nn import YOLO as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
# from learning.optimizers import AdamOptimizer as Optimizer
from learning.evaluators import RecallEvaluator as Evaluator

""" 1. Load and split datasets """
root_dir = os.path.join('/home/intern/data/coco') # FIXME
trainval_dir = os.path.join(root_dir, 'images')

# Load anchors
#anchors = dataset.load_json(os.path.join(trainval_dir, 'anchors.json'))
anchors = dataset.load_json(os.path.join('/home/intern/data/face/train', 'anchors.json'))

# Set image size and number of class
IM_SIZE = (416, 416)
NUM_CLASSES = 1

# Load trainval set and split into train/val sets
X_trainval, y_trainval = dataset.read_data(trainval_dir, IM_SIZE)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.1) # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

""" 2. Set training hyperparameters"""
hp_d = dict()

# FIXME: Training hyperparameters
hp_d['batch_size'] = 2
hp_d['num_epochs'] = 50
hp_d['init_learning_rate'] = 1e-4
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
hp_d['nms_flag'] = True

""" 3. Build graph, initialize a session and start training """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/device:GPU:1'):
  model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))

  evaluator = Evaluator()
  optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)