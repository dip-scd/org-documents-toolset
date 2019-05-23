import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfkl
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from .pickle_store import PickleStore
from .models_management import CorpusFeaturesManager

def create_keras_model(vocab_size, skipgram_range, 
                       embedding_dim = 512, hidden_dim=512):
    inp = tfkl.Input(shape=[skipgram_range*2])
    hid = inp
    embedding = tfkl.Embedding(vocab_size, embedding_dim)
    hid = embedding(hid)
    
    hid = tfkl.GRU(units=hidden_dim)(hid)

    hid = tfkl.Dense(units=vocab_size)(hid)
    hid = tfkl.Softmax()(hid)

    model = keras.Model(inputs=inp, outputs=hid)
    loss = tf.losses.sparse_categorical_crossentropy
    optimizer = tf.optimizers.Nadam()
    model.compile(loss=loss, optimizer=optimizer)
    return model

class IndexTokenizer:
    
    def __init__(self, tokenizer):
        self.word_index = OrderedDict()
        for i, word in enumerate(tokenizer.get_feature_names()):
            self.word_index[word] = i
    
    def texts_to_sequences(self, texts):
        ret = []
        for text in texts:
            row = []
            words = text.split()
            row = [
                self.word_index[w] for w in words \
                if w in self.word_index
            ]
            ret.append(row)
        return np.array(ret)
    
def prepare_training_data(corpus_texts, tokenizer, skipgram_range):
    
    def partition(lst, part_size, step=1):
        ret = []
        for i in range(0, len(lst) - part_size + 1, step):
            ret.append(lst[i:i+part_size])

        return ret

    def skipgram(lst):
        mid = int(len(lst) / 2)
        return (lst[mid], lst[:mid]+lst[mid+1:])

    def text_corpus_skipgrams(tokenizer, corpus_texts, skipgram_range):
        skipgram_size = 1 + skipgram_range * 2
        corpus_tokenized = tokenizer.texts_to_sequences(corpus_texts)
        ret = []
        for tokens_list in corpus_tokenized:
            parts = partition(tokens_list, skipgram_size, 1)
            ret += [skipgram(p) for p in parts]
        return ret
    
    train_data = text_corpus_skipgrams(tokenizer, corpus_texts, skipgram_range)
    train_x = np.array([t[1] for t in train_data])
    train_y = np.array([t[0] for t in train_data])
    return train_x, train_y

def comparison_matrix(vals, similarity_f, 
                      self_comparison_val = None,
                     symmetric = True):
    vals_ln = len(vals)
    mtx = np.zeros([vals_ln, vals_ln])
    for y in range(vals_ln):
        if symmetric:
            xrange = y
        else:
            xrange = vals_ln
        for x in range(xrange):
            sim = similarity_f(vals[y], vals[x])
            mtx[y, x] = sim
            if symmetric:
                mtx[x, y] = sim
        if self_comparison_val is None:
            mtx[y, y] = similarity_f(vals[y], vals[y])
        else:
            mtx[y, y] = self_comparison_val
    return mtx

def nonzero_indexes(vec):
        ret = []
        for i, e in enumerate(vec):
            if e != 0:
                ret.append(i)
        return ret
    
def smooth_tfidf(tfidf_, similarity_matrix):
    def cut(x):
        if np.abs(x) >= cut_val:
            return x
        else:
            return 0.
    indexes = nonzero_indexes(tfidf_)
    ret = np.zeros(len(tfidf_))
    for p in zip(indexes, tfidf_[indexes]):
        row = similarity_matrix[p[0]].copy()
        cut_val = np.max(row) * .4
        row = np.where(np.abs(row) >= cut_val, row, 0.)
        
        row *= p[1]
        ret += row

    ret /= np.linalg.norm(ret, 2)
    return ret

def cosine_sim(v1, v2):
    n1 = np.linalg.norm(v1, 2)
    n2 = np.linalg.norm(v2, 2)
    return np.dot(v1, v2) / (n1 * n2)
    

class SmoothTFIDFCorpusFeaturesManager(CorpusFeaturesManager):

    def __init__(self, path):
        super().__init__(path)
        self.model_path = path+'.model'
        self.tokenizer_store = PickleStore(path+'.tokenizer')
        self.index_tokenizer_store = PickleStore(path+'.index_tokenizer')
        self.embeddings_store = PickleStore(path+'.embeddings')
        self.sim_matrix_store = PickleStore(path+'.sim_matrix')

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.tokenizer = self.tokenizer_store.get()
        self.index_tokenizer = self.index_tokenizer_store.get()
        self.embeddings = self.embeddings_store.get()
        self.sim_matrix = self.sim_matrix_store.get()
    
    def save_model(self):
        self.sim_matrix_store.set(self.sim_matrix)
        self.embeddings_store.set(self.embeddings)
        self.index_tokenizer_store.set(self.index_tokenizer)
        self.tokenizer_store.set(self.tokenizer)
        keras.models.save_model(self.model, self.model_path)

    def fit_model(self, corpus_texts, reset = True, 
                  epochs=10,
                 batch_size=4096):
        skipgram_range = 15
        max_tokens = 10000
        
        if reset: 
            self.tokenizer = TfidfVectorizer(max_features=max_tokens)
            self.tokenizer.fit(corpus_texts)
            self.index_tokenizer = IndexTokenizer(self.tokenizer)
                                    
            model = create_keras_model(len(self.index_tokenizer.word_index),
                                      skipgram_range)
        else:
            model = self.model
            
        train_x, train_y = prepare_training_data(corpus_texts,
                                                self.index_tokenizer,
                                                skipgram_range)
        model.fit(x=train_x, y=train_y, 
                  epochs=epochs, batch_size=batch_size)
        
        self.model = model
        emb = self.model.layers[1]
        self.embeddings = emb.embeddings.numpy()
        
        self.sim_matrix = comparison_matrix(self.embeddings, 
                                         cosine_sim,
                                        self_comparison_val = 1.)
            
    def get_text_features(self, text):
        features = self.tokenizer.transform([text])
        features = np.array(features.todense().tolist()[0])
        features = smooth_tfidf(features, self.sim_matrix)
        return features