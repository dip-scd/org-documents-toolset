import numpy as np
from pickle_store import PickleStore
from sklearn.feature_extraction.text import TfidfVectorizer

class CorpusFeaturesManager:

    def __init__(self, path):
        self.model_path = path
        self.model = None

    def update(self, corpus):
        corpus_texts = corpus['values'].values
        self.model = self.fit_model(corpus_texts)
        self.save_model(self.model)

    def load(self):
        self.model = self.load_model()

    def compare(self, text, corpus):
        if self.model is None:
            self.load()

        corpus_keys = corpus.index
        corpus_texts = corpus['values'].values

        text_features = self.get_text_features(text)
        corpus_features = self.get_texts_features(corpus_texts)

        scores = np.dot(text_features, corpus_features.T)

        ret = list(zip(corpus_keys, scores))
        ret.sort(key=lambda x: -x[1])
        return ret

    def save_model(self, model):
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
        return self.store.get()

    def fit_model(self, corpus_texts):
        model = TfidfVectorizer(max_features=10000)
        model.fit(corpus_texts)
        return model

    def save_model(self, model):
        self.store.set(model)

    def get_text_features(self, text):
        features = self.model.transform([text])
        features = np.array(features.todense().tolist()[0])
        return features

    def get_texts_features(self, texts):
        features = self.model.transform(texts)
        features = np.array(features.todense())
        return features