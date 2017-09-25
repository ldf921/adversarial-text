import tflearn
import csv
import numpy as np
import os
import json

MAX_LENGTH =1014

import numpy as np
from six.moves import cPickle
class Alphabet:
    characters = '''abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n'''
    def __init__(self):
        self.lookup = np.zeros(256, dtype = np.int32)
        for i, c in enumerate(self.characters):
            self.lookup[ord(c)] = i + 1
        self.reverse_lookup = np.zeros(self.vocab_size + 1, dtype = np.uint8)
        for i, c in enumerate(self.characters):
            self.reverse_lookup[i + 1] = ord(c)
        self.reverse_lookup[0] = ord(' ')
    
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
    
    def reverse(self, char_indices):
        return np.take(self.reverse_lookup, char_indices).tobytes().strip()
    
    def reverse_raw(self, char_indices):
        return np.take(self.reverse_lookup, char_indices).tobytes()
        
    @property
    def vocab_size(self):
        return len(self.characters)
    
def onehot(arr, dims):
    return np.take(np.identity(dims), arr, axis = 0)


def clean_str(s):
    return s.replace('\\n\\n', '\n').replace('\\n','\n').replace('\\\\', '\\').replace('\\"','"').replace('<br /><br />', '\n')

def labels_onehot(labels):
    class_names = list(sorted(set(labels)))
    num_classes = len(class_names)
    class_id = dict(zip(class_names, range(num_classes)))
    y = onehot(np.array([class_id[label] for label in labels]), num_classes )
    return y

def load_dataset_csv(filepath, target_column=0):
    if filepath.find('aclImdb') != -1:
        target_column = 1
        
    alphabet = Alphabet()
    if not os.path.exists(filepath):
        filepath = filepath.replace('csv', 'json')
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        texts = dataset['texts']
        labels = dataset['labels']
    else:
        texts, labels = tflearn.data_utils.load_csv(filepath, target_column=target_column, has_header=False)
        texts[:] = [ clean_str(' '.join(text) ) for text in texts]
    
    x = tflearn.data_utils.pad_sequences([alphabet.transform(texts[i] ) for i in range(len(texts))], maxlen=1014)
    y = labels_onehot(labels)
    print('Data from {} loaded.'.format(filepath) )
    return x, y

def load_dataset_csv_raw(filepath, target_column=0):
    if filepath.find('aclImdb') != -1:
        target_column = 1
    if not os.path.exists(filepath):
        filepath = filepath.replace('csv', 'json')
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        texts = dataset['texts']
        labels = dataset['labels']
    else:
        texts, labels = tflearn.data_utils.load_csv(filepath, target_column=target_column, has_header=False)
        texts[:] = [ clean_str(' '.join(text) ) for text in texts]
    return texts, labels