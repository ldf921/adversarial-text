import tflearn
from tflearn import layers
import tensorflow as tf
from IPython import embed
from vocabulary import Vocabulary, tokenize 
from six.moves import cPickle
import data_util
import json
import h5py
from tqdm import tqdm

class TextCNN:

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, embeddings, num_filters, l2_reg_lambda=0.0, dropout=None, bn=False):
        self.input_text = layers.input_data( (None, sequence_length), dtype=tf.int32)
        
        with tf.variable_scope('Embedding'):
            embeddings_var = tf.Variable(embeddings, name='W', dtype=tf.float32)
            embeddings_var = tf.concat([np.zeros((1, embeddings.shape[1]) ), embeddings_var[1:] ] , axis = 0)
            self.embeded_text = tf.gather(embeddings_var, self.input_text)
        
        net = self.embeded_text
        for num_filter in num_filters:
            if bn:
                # , weights_init=tflearn.initializations.uniform(minval=-0.001, maxval=0.001)
                net = layers.conv_1d(net, num_filter, 3, padding='valid', activation='linear', bias=False)
                net = layers.batch_normalization(net)
                net = layers.activation(net, 'relu')
            else:
                net = layers.conv_1d(net, num_filter, 3, padding='valid', activation='relu', bias=True, regularizer='L2', weight_decay=l2_reg_lambda)
                
        if dropout is not None:
            net = layers.dropout(net, float(dropout) )
       
        features = layers.flatten( layers.max_pool_1d(net, net.shape.as_list()[1], padding='valid') )
        self.probas = layers.fully_connected(features, num_classes, activation='softmax', regularizer='L2', weight_decay=l2_reg_lambda)
        #optimizer = tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.9, lr_decay=0.2, decay_step=1000, staircase=True)
        optimizer = tflearn.optimizers.Adam(learning_rate=0.001)
        self.train_op = layers.regression(
            self.probas, 
            optimizer=optimizer,
            batch_size=128)

class TextLSTM:

    """
    A LSTM for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, embeddings, num_filters, l2_reg_lambda=0.0, dropout=None):
        self.input_text = layers.input_data( (None, sequence_length), dtype=tf.int32)
        
        with tf.variable_scope('Embedding'):
            embeddings_var = tf.Variable(embeddings, name='W', dtype=tf.float32)
            embeddings_var = tf.concat([np.zeros((1, embeddings.shape[1]) ), embeddings_var[1:] ] , axis = 0)
            self.embeded_text = tf.gather(embeddings_var, self.input_text)
        
        net = self.embeded_text
        
        self.mask = tf.expand_dims(tf.cast(tf.not_equal(self.input_text, 0), tf.float32), axis = 2)
        if dropout is not None:
            dropout = map(float, dropout.split(',') )
        for num_filter in num_filters:
            net = layers.lstm(net, num_filter, return_seq=True, dropout=dropout)
            net = tf.transpose(tf.stack(net), (1, 0, 2) )

        features = tf.reduce_sum(net * self.mask, axis=1) / (tf.reduce_sum(self.mask, axis=1) + 1e-5)
        
        self.probas = layers.fully_connected(features, num_classes, activation='softmax', regularizer='L2', weight_decay=l2_reg_lambda)
        optimizer = tflearn.optimizers.Adam(learning_rate=0.001)
        self.train_op = layers.regression(
            self.probas, 
            optimizer=optimizer,
            batch_size=128)
        
def get_timestamp():
    return time.strftime("%b%d-%H%M", time.localtime() )

def get_preprocessor(vocab, max_length):
    preprocessor = tflearn.data_preprocessing.DataPreprocessing()
    vocab.max_sequence_length = max_length
    def func(texts):
        sequences = map(lambda s : tokenize(s)[:max_length], texts)   
        return vocab.transform(sequences)
    preprocessor.add_custom_preprocessing(func)
    return preprocessor

import data_util
import argparse
import numpy as np 
import os
import time
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('action', type=str, default='train', help='train|test')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--decay', type=float, default=2e-4)
parser.add_argument('--num_filters', type=str, default=None)
parser.add_argument('--gpu', type=str, default='')
parser.add_argument('--mem', type=float, default=0.4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--length', type=int, default=200)
parser.add_argument('--checkpoint', type=int, default=None)
parser.add_argument('--dropout', type=str, default=None)
parser.add_argument('-n' , '--epochs', type=int, default=10)
FLAGS = parser.parse_args()

net_args = {
    'l2_reg_lambda' : FLAGS.decay,
    'num_filters' : list(map(int, FLAGS.num_filters.split(','))),
    'dropout' : FLAGS.dropout
}

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu
config = tflearn.config.init_graph(gpu_memory_fraction=FLAGS.mem)
dataset_name = ''.join([s[0] for s in FLAGS.dataset.split('_')]) if '_' in FLAGS.dataset else FLAGS.dataset
runs_dir = 'word/runs_{}'.format( dataset_name )
model_dir = runs_dir + '/' + FLAGS.tag

if FLAGS.action == 'train':
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
        
    dataset_file = '{}/index_data.h5'.format(runs_dir)
    if os.path.exists(dataset_file):
        with h5py.File(dataset_file, 'r') as f:
            x_train = f['x_train'][:]
            y_train = f['y_train'][:]
            x_dev = f['x_dev'][:]
            y_dev = f['y_dev'][:]
    else:
        x, y = data_util.load_dataset_csv_raw('dataset/{}/train.csv'.format(FLAGS.dataset) )
        y = data_util.labels_onehot(y)
        if FLAGS.dataset == 'liar':
            x_train, y_train = x, y
            x_dev, y_dev = data_util.load_dataset_csv_raw('dataset/{}/valid.csv'.format(FLAGS.dataset) )
            y_dev = data_util.labels_onehot(y_dev)
        else:
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            x_shuffled = [ x[i] for i in shuffle_indices ]
            y_shuffled = y[shuffle_indices]
            # Split train/test set
            n_dev_samples = int(0.1 * len(y))
            # TODO: Create a fuckin' correct cross validation procedure
            x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
            y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    vocab_file = '{}/vocab.pkl'.format(runs_dir)
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as fi:
            vocab = cPickle.load(fi)
    else:
        x_train_tokens = [ tokenize(sample) for sample in tqdm(x_train) ]
        vocab = Vocabulary(min_freq=5)
        vocab.fit(x_train_tokens)
        with open(vocab_file, 'wb') as fo:
            cPickle.dump(vocab, fo)
        
        f = h5py.File(dataset_file, 'w')
        x_train_dataset = f.create_dataset('x_train', shape=(len(x_train), FLAGS.length), dtype=np.int32)
        y_train_dataset = f.create_dataset('y_train', shape=y_train.shape, dtype=np.int32)
        x_dev_dataset = f.create_dataset('x_dev', shape=(len(x_dev), FLAGS.length), dtype=np.int32)
        y_dev_dataset = f.create_dataset('y_dev', shape=y_dev.shape, dtype=np.int32)
        y_train_dataset[:] = y_train
        y_dev_dataset[:] = y_dev
        vocab.max_sequence_length = FLAGS.length
        x_train_dataset[:] = vocab.transform(x_train_tokens).astype(np.int32)
        x_dev_dataset[:] = vocab.transform(x_dev).astype(np.int32)
        x_train = x_train_dataset[:]
        x_dev = x_dev_dataset[:]
        f.close()
    
    print('Loaded Vocabulary with {} tokens.'.format(len(vocab.vocabulary_) ) )
           
    if not os.path.exists(dataset_file):
        f = h5py.File(dataset_file, 'w')
        x_train_dataset = f.create_dataset('x_train', shape=(len(x_train), FLAGS.length), dtype=np.int32)
        y_train_dataset = f.create_dataset('y_train', shape=y_train.shape, dtype=np.int32)
        x_dev_dataset = f.create_dataset('x_dev', shape=(len(x_dev), FLAGS.length), dtype=np.int32)
        y_dev_dataset = f.create_dataset('y_dev', shape=y_dev.shape, dtype=np.int32)
        y_train_dataset[:] = y_train
        y_dev_dataset[:] = y_dev
        vocab.max_sequence_length = FLAGS.length
        x_train_dataset[:] = vocab.transform(x_train).astype(np.int32)
        x_dev_dataset[:] = vocab.transform(x_dev).astype(np.int32)
        x_train = x_train_dataset[:]
        x_dev = x_dev_dataset[:]
        f.close()
    print('Loaded training data.')

    if FLAGS.tag.startswith('lstm'):
        cnn = TextLSTM(FLAGS.length, 
                num_classes=y_train.shape[1], 
                embeddings=vocab.embeddings,
                **net_args )
    else:
        cnn = TextCNN(FLAGS.length, 
                    num_classes=y_train.shape[1], 
                    embeddings=vocab.embeddings,
                    **net_args )
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open('{}/config.json'.format(model_dir), 'w') as fo:
        json.dump(FLAGS.__dict__, fo, indent=4, sort_keys=True)
    timestamp = get_timestamp()
    model = tflearn.DNN(cnn.train_op, 
                        tensorboard_verbose=2, 
                        tensorboard_dir='{}/logs'.format(runs_dir), 
                        checkpoint_path='{}/model'.format(model_dir, FLAGS.tag), 
                        max_checkpoints=10)
    
    model.trainer.fit({model.inputs[0]:x_train, model.targets[0]:y_train}, 
              val_feed_dicts={model.inputs[0]:x_dev, model.targets[0]:y_dev}, 
              n_epoch=FLAGS.epochs,
              snapshot_step=4000,
              show_metric=True, 
              shuffle_all=True, 
              run_id=timestamp + '_' + FLAGS.tag)
    
elif FLAGS.action in ('test', 'notebook'):
    vocab_file = '{}/vocab.pkl'.format(runs_dir)
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as fi:
            vocab = cPickle.load(fi)
            
    with h5py.File('{}/index_data.h5'.format(runs_dir), 'r+') as f:
        if 'x_test' in f.keys():
            x = f['x_test'][:]
            y = f['y_test'][:]
        else:
            texts, labels = data_util.load_dataset_csv_raw('dataset/{}/test.csv'.format(FLAGS.dataset) )    
            y = data_util.labels_onehot(labels)
            if 'y_test' not in f.keys():
                dataset_y = f.create_dataset('y_test', shape=y.shape, dtype=np.int32)
                dataset_y[:] = y
            
            vocab.max_sequence_length = FLAGS.length
            x = vocab.transform(texts)
            dataset_x = f.create_dataset('x_test', shape=x.shape, dtype=np.int32)
            dataset_x[:] = x
    
    graph = tf.Graph()
    with graph.as_default():
        if FLAGS.tag.startswith('lstm'):
            cnn = TextLSTM(FLAGS.length, 
                    num_classes=y.shape[1], 
                    embeddings=vocab.embeddings,
                    **net_args )
        else:
            cnn = TextCNN(FLAGS.length, 
                        num_classes=y.shape[1], 
                        embeddings=vocab.embeddings,
                        **net_args )
        config = tflearn.config.init_graph(gpu_memory_fraction=FLAGS.mem)
        sess = tf.Session(config = config)
        tflearn.get_training_mode() 
        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restore_variables.remove(tflearn.get_training_mode() )
        saver = tf.train.Saver(restore_variables)
    
        if FLAGS.checkpoint is None:
            checkpoint_file = tf.train.latest_checkpoint(model_dir) 
        else:
            checkpoint_file = os.path.join(model_dir, 'model-{}'.format(FLAGS.checkpoint) )
        saver.restore(sess, checkpoint_file)

        model = tflearn.DNN(cnn.train_op, session=sess)
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        
        if FLAGS.action == 'test':
            print(model.evaluate(x, y, batch_size=FLAGS.batch_size) )
        elif FLAGS.action == 'notebook':
            print('Prepare for run in notebook')