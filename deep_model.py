import tflearn
from tflearn import layers
import tensorflow as tf
from IPython import embed
import json

class VDCNN:
    ''' Very Deep CNN for text classification
    '''
    def __init__(self, max_document_length, num_classes = 2, num_characters=71, num_blocks=None, char_vec_size=16, weight_decay=2e-4):
        self.input_text = layers.input_data( (None, max_document_length) )
        self.target_label = tf.placeholder(shape=(None, num_classes), dtype=tf.float32)
        
        embeded_text = layers.embedding(self.input_text, num_characters, char_vec_size)
        
        top_feature = embeded_text
        filters = 64
        if num_blocks[0] == 0:
            self.block = (2, 2, 2, 2)
        else:
            self.block = num_blocks
        
        for i, num_block in enumerate(self.block):
            if i > 0:
                filters *= 2
                top_feature = layers.max_pool_1d(top_feature, 3, strides=2, padding='same')
            for block_i in range(num_block):
                top_feature = self.conv_block(top_feature, filters)
        
        pooled_feature = layers.flatten(layers.custom_layer(top_feature, self.kmax_pool_1d) )
        fc1 = layers.fully_connected(pooled_feature, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        fc2 = layers.fully_connected(fc1, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        self.probas = layers.fully_connected(fc2, num_classes, activation='softmax', regularizer='L2', weight_decay=weight_decay)
        self.train_op = layers.regression(self.probas, placeholder=self.target_label)
        
    @staticmethod
    def conv_block(t, filters):
        t = layers.conv_1d(t, filters, 3, activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        t = layers.conv_1d(t, filters, 3, activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        return t
    
    @staticmethod
    def kmax_pool_1d(t):
        tf.transpose(t, (0, 2, 1) )
        values, _ = tf.nn.top_k(t)
        return values

class VDCNNV2:
    ''' Very Deep CNN for text classification
        1. using (He et.al 2015) initialization, $N(0,\sqrt{2 / n_in })$
        2. using sgd with learning_rate=0.01, momentum=0.9
    '''
    def __init__(self, max_document_length, num_classes = 2, num_characters=71, char_vec_size=16, weight_decay=2e-4, optimizer='sgd', dropout=None,
                num_blocks=None):
        self.input_text = layers.input_data( (None, max_document_length) )
        self.target_label = tf.placeholder(shape=(None, num_classes), dtype=tf.float32)
        
        embeded_text = layers.embedding(self.input_text, num_characters, char_vec_size)
        mask = tf.cast(tf.not_equal(self.input_text, 0), tf.float32)
        embeded_text = embeded_text * tf.expand_dims(mask, 2)
        self.embeded_text = embeded_text
        
        top_feature = embeded_text
        filters = 64
        if num_blocks[0] == 0:
            self.block = (1, 1, 1, 1)
        else:
            self.block = num_blocks
        for i, num_block in enumerate(self.block):
            if i > 0:
                filters *= 2
                top_feature = layers.max_pool_1d(top_feature, 3, strides=2, padding='same')
            for block_i in range(num_block):
                top_feature = self.conv_block(top_feature, filters)
                
        pooled_feature = layers.flatten(layers.custom_layer(top_feature, self.kmax_pool_1d) )
        if dropout is not None:
            pooled_feature = layers.dropout(pooled_feature, dropout)
        fc1 = layers.fully_connected(pooled_feature, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        if dropout is not None:
            fc1 = layers.dropout(fc1, dropout)
        fc2 = layers.fully_connected(fc1, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        self.probas = layers.fully_connected(fc2, num_classes, activation='softmax', regularizer='L2', weight_decay=weight_decay)
        def build_sgd(learning_rate):
            step_tensor = tf.Variable(0., name="Training_step",
                                      trainable=False)
            steps = [-1.0, 16000.0, 24000.0]
            lrs = [1e-1, 1e-2, 1e-3]
            lr = tf.reduce_min(tf.cast(tf.less(step_tensor, steps), tf.float32) + lrs)
            tflearn.helpers.summarizer.summarize(lr, 'scalar', 'lr', 'Optimizer_training_summaries')
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9), step_tensor
        if optimizer == 'sgd':
            optimizer = build_sgd
        self.train_op = layers.regression(self.probas, optimizer=optimizer, learning_rate=0.001, placeholder=self.target_label)
        
    @staticmethod
    def conv_block(t, filters):
        t = layers.conv_1d(t, filters, 3, weights_init=tflearn.initializations.variance_scaling(), activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        t = layers.conv_1d(t, filters, 3, weights_init=tflearn.initializations.variance_scaling(), activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        return t
    
    @staticmethod
    def kmax_pool_1d(t):
        tf.transpose(t, (0, 2, 1) )
        values, _ = tf.nn.top_k(t)
        return values
    
class VDCNNV3:
    ''' Very Deep CNN for text classification
        1. using (He et.al 2015) initialization, $N(0,\sqrt{2 / n_in })$
        2. using sgd with learning_rate=0.01, momentum=0.9
    '''
    def __init__(self, max_document_length, num_classes = 2, num_characters=71, char_vec_size=16, weight_decay=2e-4, optimizer='sgd', dropout=None):
        self.input_text = layers.input_data( (None, max_document_length) )
        self.target_label = tf.placeholder(shape=(None, num_classes), dtype=tf.float32)
        
        embeded_text = layers.embedding(self.input_text, num_characters, char_vec_size)
        mask = tf.cast(tf.not_equal(self.input_text, 0), tf.float32)
        embeded_text = embeded_text * tf.expand_dims(mask, 2)
        self.embeded_text = embeded_text
        
        top_feature = embeded_text
        filters = 64
        self.block = (1, 1, 1, 1)
        for i, num_block in enumerate(self.block):
            if i > 0:
                filters *= 2
                top_feature = layers.max_pool_1d(top_feature, 3, strides=2, padding='same')
            for block_i in range(num_block):
                top_feature = self.conv_block(top_feature, filters)
                
        pooled_feature = layers.flatten(layers.custom_layer(top_feature, self.kmax_pool_1d) )
        #if dropout is not None:
            #pooled_feature = layers.dropout(pooled_feature, dropout)
        fc1 = layers.fully_connected(pooled_feature, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        #if dropout is not None:
        #    fc1 = layers.dropout(fc1, dropout)
        #fc2 = layers.fully_connected(fc1, 2048, activation='relu', regularizer='L2', weight_decay=weight_decay)
        self.probas = layers.fully_connected(pooled_feature, num_classes, activation='softmax', regularizer='L2', weight_decay=weight_decay)
        def build_sgd(learning_rate):
            step_tensor = tf.Variable(0., name="Training_step",
                                      trainable=False)
            steps = [-1.0, 20000.0, 32000.0]
            lrs = [1e-2, 5e-3, 1e-3]
            lr = tf.reduce_min(tf.cast(tf.less(step_tensor, steps), tf.float32) + lrs)
            tflearn.helpers.summarizer.summarize(lr, 'scalar', 'lr', 'Optimizer_training_summaries')
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9), step_tensor
        if optimizer == 'sgd':
            optimizer = build_sgd
        self.train_op = layers.regression(self.probas, optimizer=optimizer, learning_rate=0.001, placeholder=self.target_label)
        
    @staticmethod
    def conv_block(t, filters):
        t = layers.conv_1d(t, filters, 3, weights_init=tflearn.initializations.variance_scaling(), activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        t = layers.conv_1d(t, filters, 3, weights_init=tflearn.initializations.variance_scaling(), activation='linear', bias=False)
        t = layers.batch_normalization(t)
        t = layers.activation(t, 'relu')
        return t
    
    @staticmethod
    def kmax_pool_1d(t):
        tf.transpose(t, (0, 2, 1) )
        values, _ = tf.nn.top_k(t)
        return values

def get_timestamp():
    return time.strftime("%b%d-%H%M", time.localtime() )

import data_util
import argparse
import numpy as np 
import os
import time
from IPython import embed
import functools

parser = argparse.ArgumentParser()
parser.add_argument('action', type=str, default='train', help='train|test')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--gpu', type=str, default='')
parser.add_argument('--mem', type=float, default=0.4)
# parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--checkpoint', type=int, default=None)
parser.add_argument('--blocks', type=str, default="0,0,0,0")
parser.add_argument('-v', '--version', type=str, default='1')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tflearn.config.init_graph(gpu_memory_fraction=FLAGS.mem)
dataset_name = ''.join([s[0] for s in FLAGS.dataset.split('_')]) if '_' in FLAGS.dataset else FLAGS.dataset
runs_dir = 'runs_{}'.format( dataset_name )
model_dir = '{}/v{}'.format(runs_dir, FLAGS.version)

if FLAGS.version.startswith('1'):
    net_class = VDCNN
    model_dir = runs_dir
elif FLAGS.version.startswith('2'):
    net_class = VDCNNV2
elif FLAGS.version.startswith('3'):
    net_class = functools.partial(VDCNNV3, optimizer='adam', dropout=0.5)

if FLAGS.action == 'train':
    x, y = data_util.load_dataset_csv('dataset/{}/train.csv'.format(FLAGS.dataset) )    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    n_dev_samples = int(0.1 * len(y))
    # TODO: Create a fuckin' correct cross validation procedure
    x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
    y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    cnn = net_class(1014, num_classes=y.shape[1], num_blocks=map(int, FLAGS.blocks.split(',') ) )
    timestamp = 'v{}_{}'.format(FLAGS.version, get_timestamp())
    model = tflearn.DNN(cnn.train_op, 
                        tensorboard_verbose=0, 
                        tensorboard_dir='{}/logs'.format(runs_dir), 
                        checkpoint_path='{}/model'.format(model_dir), 
                        max_checkpoints=5)
    
    with open('{}/config.json'.format(model_dir), 'w') as fo:
        json.dump(FLAGS.__dict__, fo, indent=4, sort_keys=True)
    
    snapshot_step = min(x_train.shape[0] // 128, 4000)
    model.fit(x_train, y_train, validation_set=(x_dev, y_dev), batch_size=128, 
              n_epoch=10,
              show_metric=True, 
              shuffle=True, 
              run_id=timestamp,
              snapshot_step=snapshot_step, 
              snapshot_epoch=False)
    
elif FLAGS.action in ('test', 'notebook'):
    x, y = data_util.load_dataset_csv('dataset/{}/test.csv'.format(FLAGS.dataset) )    
    
    graph = tf.Graph()
    with graph.as_default():
        config = tflearn.config.init_graph(gpu_memory_fraction=FLAGS.mem)
        cnn = net_class(1014, num_classes=y.shape[1], num_blocks=map(int, FLAGS.blocks.split(',') ) )
        
        sess = tf.Session(config = config)
        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restore_variables.remove(tflearn.get_training_mode() )
        saver = tf.train.Saver(restore_variables)
        if FLAGS.checkpoint is None:
            checkpoint_file = tf.train.latest_checkpoint(model_dir)
        else:
            checkpoint_file = '{}/model-{}'.format(model_dir, FLAGS.checkpoint)
        #print(model_dir)
        saver.restore(sess, checkpoint_file)

        model = tflearn.DNN(cnn.train_op, session=sess)

        if FLAGS.action == 'test':
            print(model.evaluate(x, y, batch_size=FLAGS.batch_size) )
        elif FLAGS.action == 'notebook':
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            print('Prepare for run in notebook')