from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix
import numpy as np
import h5py
import os
from six.moves import cPickle
import collections

def sparse_onehot(m, vocab):
    cols = []
    rows = []
    
    for i in range(m.shape[0]):
        words = np.unique(m[i])
        cols.append(words)
        rows.append([i, ] * len(words) )
    
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    
    return csr_matrix( (np.ones(len(rows) ), (rows, cols) ), shape=(m.shape[0], len(vocab.vocabulary_) ) )

def load_dataset(dataset):
    dataset_name = ''.join([s[0] for s in dataset.split('_')]) if '_' in dataset else dataset
    runs_dir = 'word/runs_{}'.format( dataset_name )
    dataset_file = '{}/index_data.h5'.format(runs_dir)
    with h5py.File(dataset_file, 'r') as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_test)))
    x_shuffled = x_test[shuffle_indices]
    y_shuffled = y_test[shuffle_indices]
    vocab_file = '{}/vocab.pkl'.format(runs_dir)
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as fi:
            vocab = cPickle.load(fi)
    print('Dataset {} loaded ..'.format(dataset) )
    return x_train, y_train, x_shuffled, y_shuffled, vocab
            
def model(x_train, y_train, x_test, y_test, vocab):
    clf = MultinomialNB()
    clf.fit( sparse_onehot(x_train, vocab), np.argmax(y_train, axis = -1) )
    print(clf.score( sparse_onehot(x_test, vocab) , np.argmax(y_test, axis = -1) ) )
    return clf


import solver
import numpy as np
#import cv2
from IPython.display import display
import ipywidgets
import kenlm
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
    
class WordVector:
    def __init__(self, embeddings, vocab):
        self.embeddings = embeddings
        self.embeddings_unit_vec = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        self.vocab = vocab
        self._most_similar_cache = dict()
    
    def most_similar(self, word_idx, num=20):
        if np.any(self.embeddings_unit_vec[word_idx]):
            ret = self._most_similar_cache.get( (word_idx, num) )
            if ret is None:
                s = np.dot(self.embeddings_unit_vec, self.embeddings_unit_vec[word_idx])
                top_indices = np.argpartition(s, -num)[-num:]
                ret = [ (idx, s[idx]) for idx in top_indices ]
                self._most_similar_cache[(word_idx, num)] = ret
            return ret
        else:
            return []
        
    def most_similar_word(self, word, num=20):
        return [ ( unidecode(self.vocab.reverse(idx).decode('utf8') ) , score) for idx, score in 
                self.most_similar(self.vocab.get(word), num=num) ]
    
    _cache = dict()
    
    @classmethod
    def load(cls, fname):
        ret = cls._cache.get(fname)
        if ret is None:
            with open(fname, 'r') as fi:
                tokens = map(lambda line : line.rstrip().split(' '), fi.readlines() ) 
                vocab = WordIndex(map(lambda t : t[0], tokens), oov=None)
                embeddings = None
                for token in tokens:
                    if embeddings is None:
                        vector_size = len(token) - 1
                        embeddings = np.zeros([len(vocab), vector_size])
                    embeddings[vocab.get(token[0])] = list(map(float, token[1:]) )
            ret = cls(embeddings, vocab)
            cls._cache[fname] = ret
        return ret        
import nltk
from copy import copy 
import ipywidgets
from IPython.display import display
import re
from unidecode import unidecode

class Printer:
    def __init__(self):
        self.widget = ipywidgets.HTML()
        self.value = ''
        display(self.widget)
        
    def print(self, *args, **kwargs):
        end = kwargs.get('end', '<br/>')
        self.value += u' '.join(map(unicode, args) ) + end
        self.widget.value = self.value

def is_punct(s):
    for ch in s.lower():
        i = ord(ch)
        if 97 <= i <= 122 or 48 <= i <= 57:
            return False
    return True

def parse_tokens_lm(tokens):
    parsed_tokens = [ token.lower() for token in tokens if not is_punct(token) ]
    return ' '.join(parsed_tokens)

def unigram_lm_scores(original_tokens, p, lm):
    def scorer(substitution):
        return [lm.score(substitution, eos=False, bos=False)]
    return scorer

def local_lm_scores(original_tokens, p, lm, order=4, printer=None):
    
    s = p
    while s >= 0:
        if re.match(r'[\.!?]+', original_tokens[s]):
            break
        s -= 1
    
    tokens = ['|--']
    bos=True
    
    if s >= p - order + 1:
        s += 1
    else:
        s = p-order+1
    
    e = p
    while e < len(original_tokens):
        if re.match(r'[\.!?!]+', original_tokens[e]):
            break
        e += 1
    
    eos=True
   
    if e <= p + order - 1:
        eos=True
    else:
        eos=False
        e = p + order
    
    target_index = None
    for i in range(s, e):
        if not is_punct(original_tokens[i]):
            tokens.append(original_tokens[i])
            if i == p:
                target_index = len(tokens) - 1
                
    tokens.append('--|')
    
    if printer is not None:
        shift = int(bos)
        for i, (score, ngram, oov) in enumerate(lm.full_scores(' '.join(tokens[1:-1] ), eos=eos, bos=bos)):
            if i >= target_index - 1:
                printer.print('{:.2f} {}'.format(score, ' '.join(tokens[i-ngram+1+shift:i+1+shift]) ), end=' ')
        printer.print()
    
    original_pos = nltk.pos_tag(tokens[1:-1])[target_index-1]
    
    def evaluate_score(substitution=None):
        if target_index is not None:
            tokens[target_index] = substitution
            
            for i, (score, ngram, oov) in enumerate(lm.full_scores(' '.join(tokens[1:-1]), bos=bos, eos=eos)):
                if i >= target_index - 1 and i < target_index - 1 + order:
                    yield score    
    return evaluate_score

def join_tokens(tokens):
    s = tokens[0]
    for i in range(1 ,len(tokens) ):
        if re.match(r'''^n't|'[a-z]+|[^\(`\$a-z0-9]+$''', tokens[i]) is None and re.match(r'\(|``|\$', tokens[i-1]) is None:
            s += ' '
        format_token = tokens[i].replace('``','"').replace("''", '"')
        s += format_token
    return s

import scipy
from tqdm import tqdm_notebook
def bag_of_word(texts, vocab):
    x = scipy.sparse.dok_matrix( (len(texts), len(vocab.vocabulary_) ) )
    if len(texts) > 1000:
        for i, tokens in tqdm_notebook(enumerate(texts) ):
            for token in tokens[:2000]:
                x[i, vocab.vocabulary_.get(token)] = 1
    else:
        for i, tokens in enumerate(texts):
            for token in tokens[:2000]:
                x[i, vocab.vocabulary_.get(token)] = 1
    return x

from vocabulary import Vocabulary
def preprocess(vocab, token_sequences, maxlen=1014):
    if isinstance(vocab, Vocabulary):
        #return vocab.transform(token_sequences)
        return bag_of_word(token_sequences, vocab)
    else:
        ret = np.zeros( (len(token_sequences), maxlen) )
        for i, token_sequence in enumerate(token_sequences):
            sequence = join_tokens(token_sequence)
            indices = vocab.transform(sequence[:maxlen])
            ret[i, :len(indices)] = indices
        return rets

from copy import copy
    
def solve_greedy_search(model, vocab, sequence, target, lm, verbose=0, target_proba=0.8, p=None, lm_loss_limit=4.0, unigram=False, 
                        target_diffs=None,
                        max_sub=4,
                       max_steps=15):
    # wordvec = WordVector(embeddings)
    wordvec = WordVector.load('../word-embedding/counter-fitting/word_vectors/counter-fitted-vectors.txt')
    
    if p is None:
        p = Printer()
    
    if isinstance(vocab, Vocabulary):
        tokens = [ vocab.vocabulary_.reverse(word_idx) for word_idx in np.trim_zeros(sequence) ]
        #tokens = sequence
    else:
        text = vocab.reverse(sequence)
        tokens = nltk.word_tokenize(text)
    
    candidates = dict([(i, wordvec.most_similar_word(token) ) for i, token in enumerate(tokens) if wordvec.vocab.get(token) is not None ] )
    # candidates = dict([(i, wordnet_similar(vocab, word_idx) ) for i, word_idx in enumerate(sequence) if word_idx != 0 ])
    bow_feature = preprocess(vocab, [tokens]) 
    score = np.dot(model.predict_proba(bow_feature), target)[0]
    original_score = score
    p.print('score={:.2f} Adversarial Target {}'.format(score, ['NEG', 'POS'][np.argmax(target)]) )
    #base_lm_score = lm.score(decode(sequence, vocab))
    #print('score={:.2f}, lm={:2f}, {}'.format(score, lm.score(decode(sequence, vocab)), decode(sequence, vocab) ) )
    keywords = ["and", "the", "a", "but"]
    #keywords = set([ vocab.vocabulary_.get(keyword) for keyword in keywords])
    original_tokens = tokens
    
    contrib_scores = np.transpose(np.array([model.feature_log_prob_[0] - model.feature_log_prob_[1], 
                               model.feature_log_prob_[1] - model.feature_log_prob_[0]]) )
    
    
    if target_diffs is not None:
        target_diffs = np.round( target_diffs * len(tokens) )
        #target_proba = 2.0
    else:
        target_diffs = 1000
    if not isinstance(target_proba, collections.Iterable):
        target_proba = (target_proba, )
    target_proba = list(sorted(target_proba))
    ret = [ {'score' : 0, 
            'diff' : 0, 
            'tokens' : [],
            'original_score' : original_score,
            'original_tokens' : original_tokens,
            'lm_loss_limit' : lm_loss_limit,
            'target' : target} for _ in target_proba ]
    diffs = 0
    for steps in range(max_steps):
        prev_sequence = tokens
        prev_score = score
        next_sequences = []
        lm_losses = []
        substitutions = []
        predictions = []
        
        for i in candidates:
            if unigram:
                scorer = unigram_lm_scores(tokens, i, lm)
            else:
                scorer = local_lm_scores(tokens, i, lm)
            
            orig_score = np.array(list(local_lm_scores(original_tokens, i, lm)(original_tokens[i])))
            
            for w, similarity in candidates[i]:
                if tokens[i] not in keywords:
                    next_sequence = copy(tokens)
                    next_sequence[i] = w
                    next_sequences.append(next_sequence)
                    # lm_losses.append( np.sum(np.maximum(0, orig_score - list(scorer(w) ) ) ** 2) )
                    lm_losses.append( max(0, np.max(orig_score - list(scorer(w) ) ) ) )
                    
                    delta_feature = -contrib_scores[vocab.vocabulary_.get(tokens[i])]
                    if bow_feature[0, vocab.vocabulary_.get(w)] == 0:
                        delta_feature += contrib_scores[vocab.vocabulary_.get(w)]
                    if tokens[i] == w:
                        delta_feature = [0, 0]
                        
                    predictions.append( -np.dot(delta_feature, target) )
                    
                    # lm_losses.append(0)
                    substitutions.append(i)
                    
        #features = preprocess(vocab, next_sequences)
        lm_losses = np.array(lm_losses)
        objectives = predictions + (lm_losses > lm_loss_limit).astype(np.float32) * 1000
        
        substituted = set()
        tokens = copy(tokens)
        subs = 0
        for i in np.argsort(objectives):
            sp = substitutions[i]
            if sp not in substituted:
                substituted.add(sp)
                diffs -= (tokens[sp] != original_tokens[sp])
                tokens[sp] = next_sequences[i][sp]
                diffs += (tokens[sp] != original_tokens[sp])
                
                subs += 1
                bow_feature = preprocess(vocab, [tokens])
                score = np.dot( model.predict_proba(bow_feature ), target )[0]
                
                f = True
                for i, tp in enumerate(target_proba):
                    if score >= tp:
                        if not ret[i]['tokens']:
                            ret[i]['score'] = score 
                            ret[i]['diff'] = diffs
                            ret[i]['tokens'] = tokens
                    else:
                        f = False
                        break
                if f:
                    break
                
                if subs >= max_sub or diffs >= target_diffs:
                    break
    
        if f:
            break
        if score < prev_score or diffs >= target_diffs:
            break
                
        if verbose >= 1: 
            p.print(u'score={:.2f} lm={:.2f}'.format(score, lm.score(parse_tokens_lm(tokens) ) ) )
            #p.print(parse_tokens_lm(tokens) )
            for t, ot in zip(tokens, original_tokens):
                if ot != t:
                    p.print('<del>' + ot + '</del>', end=' ')
                    p.print('<b>' + t + '</b>', end = ' ')
                else:
                    p.print(t, end=' ')
            p.print()
            # p.print(join_tokens(tokens)[:1014] )
        
    diff = 0 
    p.print(u'score={:.2f} lm={:.2f}'.format(score, lm.score(parse_tokens_lm(tokens) ) ) )
    for t, ot in zip(tokens, original_tokens):
        if ot != t:
            p.print('<del>' + ot + '</del>', end=' ')
            p.print('<b>' + t + '</b>', end = ' ')
            p.print('({:.3f} {:.3f})'.format(np.dot(target, contrib_scores[vocab.vocabulary_.get(ot)]), 
                                  np.dot(target, contrib_scores[vocab.vocabulary_.get(t)])) , end=' ')
            diff += 1
        else:
            p.print(t, end=' ')
    p.print()
    for i, tp in enumerate(target_proba):
        if not ret[i]['tokens']:
            ret[i]['score'] = score 
            ret[i]['diff'] = diffs
            ret[i]['tokens'] = tokens
    return ret    

    

def compare(tokens, original_tokens, p):
    def encode_html(s):
        return s.replace('<', '&lt;').replace('>', '&gt;')
    for t, ot in zip(tokens, original_tokens):
        if ot != t:
            p.print('<del>' + encode_html(ot) + '</del>', end=' ')
            p.print('<b>' + encode_html(t) + '</b>', end = ' ')
        else:
            p.print(encode_html(t), end=' ')
            
class FilePrinter:
    def __init__(self, fname):
        self.f = open(fname, 'w')
        
    def print(self, *args, **kwargs):
        end = kwargs.get('end', '\n')
        self.f.write(' '.join(map(str, args) ) + end)
        
    def close(self):
        self.f.close()
        
class NullPrinter:
    def print(self, *args, **kwargs):
        pass
        
def full_scores(s, printer):
    for sent in nltk.sent_tokenize(s):
        tokens = ['<s>'] + filter(lambda x : not is_punct(x), nltk.word_tokenize(sent) ) + ['</s>']
        printer.print(sent)
        for i, (score, ngram, oov) in enumerate(lm.full_scores(' '.join(tokens[1:-1]), eos=True, bos=True)):
            printer.print('{:.2f} {:.2f} {} {}'.format(score, score - lm.score(tokens[i+1], bos=False, eos=False), ' '.join(tokens[i+2-ngram: i+2]), ngram) )
            
    
from tqdm import tqdm_notebook
import numpy as np

class FilePrinter:
    def __init__(self, fname):
        self.f = open(fname, 'w')
        
    def print(self, *args, **kwargs):
        end = kwargs.get('end', '\n')
        self.f.write(' '.join(map(str, args) ) + end)
        
    def close(self):
        self.f.close()

def main(clf, data, samples, lm_loss_limit=2.0, printer=None, **kwargs):
    results = []
    if isinstance(samples, int):
        samples = range(samples)
    for i in tqdm_notebook(samples):
        results.extend(solve_greedy_search(clf, data.vocab, data.x[i], 1-data.y[i], p=printer or NullPrinter(), lm=data.lm, 
                                                 lm_loss_limit=lm_loss_limit, **kwargs) )
    return results    