from nltk import word_tokenize
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
import collections

class WordIndex:
    def __init__(self, words, oov):
        self.words = words
        self.oov = oov
        self.indices = dict(list(zip(words, range(len(words) ) ) ) )
    
    @staticmethod
    def parse_word(key):
        return key.lower()
    
    def get(self, key):
        return self.indices.get(self.parse_word(key), self.oov) 
    
    def reverse(self, index):
        return self.words[index]
    
    def __len__(self):
        return len(self.words)
        
    
class Vocabulary:
    def __init__(self, min_freq = 10):
        self.min_freq = min_freq
        self.oov = 1

    def fit(self, sequences, embedding=True):
        freq = dict()
        for sequence in sequences:
            for token in sequence:
                t = token.lower()
                if t not in freq:
                    freq[t] = 0
                freq[t] += 1
        valid_tokens = ['<pad>', '<oov>'] + [k for k, v in  freq.iteritems() if v > self.min_freq ]
        self.vocabulary_ = WordIndex(valid_tokens, 1)
        print('Total tokens = {}'.format(len(valid_tokens) ) )
        if embedding:
            self.get_embeddings()
            print('Word2vec embeddings built ...')
        
    def transform(self, sequences):
        ret = np.zeros([len(sequences), self.max_sequence_length])
        gen = enumerate(sequences)
        if len(sequences) > 10000:
            gen = tqdm(gen)
        for i, sequence in gen:
            if isinstance(sequence, list):
                tokens = sequence[:self.max_sequence_length]
            else:
                tokens = tokenize(sequence)[:self.max_sequence_length]
            for j, token in enumerate(tokens):
                ret[i, j] = self.vocabulary_.get(token)
        return ret
    
    def reverse(self, sequences):
        if not isinstance(sequences[0], collections.Iterable):
            sequences = [sequences]
        return [' '.join([self.vocabulary_.reverse(word_idx) for word_idx in sequence if word_idx != 0 ]) for sequence in sequences ]
        
    
    def get_embeddings(self):
        import gensim
        vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        ret = np.zeros([len(self.vocabulary_), vec.vector_size])
        for i in range(len(self.vocabulary_) ):
            try:
                ret[i] = vec.word_vec(self.vocabulary_.reverse(i) )
            except KeyError:
                pass
        self.embeddings = self.random_initialize(ret)
        
    @staticmethod
    def random_initialize(embeddings):
        mask = np.any(embeddings, axis = -1).astype(np.float32)
        shape = embeddings.shape
        avg_norm = np.sum(np.linalg.norm(embeddings, axis = -1) ) / np.sum(mask)
        a = 3 * avg_norm / (shape[-1] ** 0.5)
        return np.where(
            np.any(embeddings, axis = -1, keepdims=True), 
            embeddings, 
            np.concatenate([ np.zeros( (1, shape[1]) ), 
                       np.random.uniform(-a, a, (shape[0] - 1, shape[1]) )], axis = 0) )
    
def tokenize(sequence):
    if isinstance(sequence, unicode):
        sequence = unidecode(sequence)
    return word_tokenize(sequence)