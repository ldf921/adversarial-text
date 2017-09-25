import csv
import numpy as np
from keras import backend as K
samples = {}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
MAX_LENGTH =1014

import numpy as np
from six.moves import cPickle
class Alphabet:
    characters = '''abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n'''
    def __init__(self):
        self.lookup = np.zeros(256, dtype = np.int32)
        for i, c in enumerate(self.characters):
            self.lookup[ord(c)] = i + 1
    
    def transform(self, s):
        ''' Return the index of characters in ``s``, for characters in alphabet, return index from 1, 
        from characters out of alphabet, return index 0, transform upper case to lower case
        Args:
            s (string) 
        Return:
            array(dtype = np.int32) of same length of ``s``
        '''
        arr = np.frombuffer(s.lower(), dtype=np.uint8)
        return np.take(self.lookup, arr)
    
    @property
    def vocab_size(self):
        return len(self.characters)
    
def onehot(arr, dims):
    return np.take(np.identity(dims), arr, axis = 0)
   
def load_data(filepath, max_document_length):
    alphabet = Alphabet()
    with open(filepath, 'rb') as fi:
        texts, labels = cPickle.load(fi)
        y = onehot(labels, 2)
        x = np.zeros([len(texts), max_document_length], dtype=np.int32)
        for i, text in enumerate(texts):
            text_arr = alphabet.transform(text[:max_document_length])
            x[i, : len(text_arr)] = text_arr
        return x, y    

x, y = load_data('../data/aclImdb/chr-imdb-train.pkl', MAX_LENGTH)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
n_dev_samples = 1000
# TODO: Create a fuckin' correct cross validation procedure
x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

import model_keras
import model
import tensorflow as tf
import os
import time
import datetime
def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch-1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch = x_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(x_batch, y_batch))
            yield batch
            
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--gpu', type=str, default="")
CharCNN = model.CharCNN
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
batch_size = args.batch_size
num_epochs = args.num_epochs

with tf.Graph().as_default():
    cnn = CharCNN(num_classes=2)
    session_conf = tf.ConfigProto(
        log_device_placement=False)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # print('updates oprs = {}'.format(len(cnn.updates.updates)))
        if hasattr(cnn, 'updates'):
            with tf.control_dependencies(cnn.updates):
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        else:
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "runs_new", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already
        # exists, so we need to create it.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        # Initialize all variables
        sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            x_batch = np.array(x_batch)
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.training:1,
              cnn.dropout_keep_prob: 0.5,
              K.learning_phase() : 1
            }       
            
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,
                    cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # write fewer training summaries, to keep events file from
            # growing so big.
            if step % (evaluate_every / 2) == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
        def dev_step(x_batch, y_batch, writer=None):
            dev_size = len(x_batch)
            max_batch_size = 500
            num_batches = dev_size/max_batch_size
            acc = []
            losses = []
            #print("Number of batches in dev set is " + str(num_batches))
            for i in range(num_batches):
                x_batch_dev = x_batch[i * max_batch_size:(i + 1) * max_batch_size]
                y_batch_dev = y_batch[i * max_batch_size: (i + 1) * max_batch_size]
                feed_dict = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                  cnn.training:0,
                  cnn.dropout_keep_prob: 1.0,
                  K.learning_phase() : 0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                acc.append(accuracy)
                losses.append(loss)
                time_str = datetime.datetime.now().isoformat()
                #print("batch " + str(i + 1) + " in dev >>" +" {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            print("Mean accuracy=" + str(sum(acc)/len(acc)))
            print("Mean loss=" + str(sum(losses)/len(losses)))
        # just for epoch counting
        num_batches_per_epoch = int(len(x_train)/batch_size) + 1
        # Generate batches
        batches = batch_iter(x_train, y_train,batch_size, num_epochs)
        evaluate_every = 300
        checkpoint_every =900
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("Epoch: {}".format(
                    int(current_step / num_batches_per_epoch)))
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(
                    sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))