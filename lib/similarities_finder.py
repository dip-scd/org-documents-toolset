class SimilaritiesFinder:

    def __init__(self, comparer, max_similarities, similarity_threshold = 0.):
        self.comparer = comparer
        self.max_similarities = max_similarities
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def refine_similarities_list(lst):
        lst = list(set(lst))
        lst = [k for k in lst if k[0] != k[1]]
        return lst

    @staticmethod
    def refine_symmetric_similarities_list(lst):
        lst = [(min(k), max(k)) for k in lst]
        return SimilaritiesFinder.refine_similarities_list(lst)

    def find_similarities(self, text, target_corpus):
        most_similar = self.comparer.compare(text, target_corpus)
        top_keys = \
            [r[0] for r in most_similar[:self.max_similarities] if r[1] >= self.similarity_threshold]
        return top_keys

    def find_similarities_between_lists(self, source_corpus, target_corpus):
        ret = []
        for key in source_corpus.index:
            text = source_corpus.loc[key]['values']
            similar_keys = self.find_similarities(text, target_corpus)
            for target_key in similar_keys:
                ret.append((key, target_key))
        if list(source_corpus.index) == list(target_corpus.index):
            ret = SimilaritiesFinder.refine_symmetric_similarities_list(ret)
        else:
            ret = SimilaritiesFinder.refine_similarities_list(ret)
        return ret