import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .pickle_store import PickleStore

class CorpusFeaturesManager:

    def __init__(self, path):
        self.model_path = path
        self.model = None

    def update(self, corpus, *args, **kwargs):
        corpus_texts = corpus['values'].values
        self.fit_model(corpus_texts, *args, **kwargs)
        self.save_model()
        
    def calculate_scores(self, text_features, corpus_features):
        return np.dot(text_features, corpus_features.T)

    def compare(self, text, corpus):
        if self.model is None:
            self.load_model()
            
        corpus_keys = corpus.index
        
        if len(corpus) > 0:
            corpus_texts = corpus['values'].values

            text_features = self.get_text_features(text)
            corpus_features = self.get_texts_features(corpus_texts)

            scores = self.calculate_scores(text_features, corpus_features)

            ret = list(zip(corpus_keys, scores))
            ret.sort(key=lambda x: -x[1])
            return ret
        else:
            return []

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def get_texts_features(self, texts):
        return np.array([self.get_text_features(t) for t in texts])

    def get_text_features(self, text):
        raise NotImplementedError

    def fit_model(self, corpus_texts):
        raise NotImplementedError


class TFIDFCorpusFeaturesManager(CorpusFeaturesManager):

    def __init__(self, path):
        super().__init__(path)
        self.store = PickleStore(path)

    def load_model(self):
        self.model = self.store.get()

    def fit_model(self, corpus_texts):
        self.model = TfidfVectorizer(max_features=10000)
        self.model.fit(corpus_texts)

    def save_model(self):
        self.store.set(self.model)

    def get_text_features(self, text):
        features = self.model.transform([text])
        features = np.array(features.todense().tolist()[0])
        return features

    def get_texts_features(self, texts):
        features = self.model.transform(texts)
        features = np.array(features.todense())
        return features