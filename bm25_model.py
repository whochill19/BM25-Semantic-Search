# bm25_module.py
import math
import numpy as np
from collections import Counter
import pickle

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = None
        self.doc_lengths = None
        self.avg_doc_length = 0
        self.doc_freqs = None
        self.idf = None
        self.vocab = None

    def fit(self, documents):
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = np.mean(self.doc_lengths)

        self.vocab = set()
        self.doc_freqs = Counter()

        for doc in documents:
            tokens = doc.split()
            self.vocab.update(tokens)
            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.idf = {}
        N = len(documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        print("BM25 model trained successfully!")
        print(f"Vocabulary size        : {len(self.vocab)} terms")
        print(f"Average document length: {self.avg_doc_length:.1f} words\n")

        return self

    def _score_document(self, query_terms, doc_index):
        score = 0
        doc_tokens = self.documents[doc_index].split()
        doc_length = self.doc_lengths[doc_index]

        term_freqs = Counter(doc_tokens)

        for term in query_terms:
            if term not in self.vocab:
                continue

            # perhitungan frekuensi kata dalam satu dokumen
            tf = term_freqs.get(term, 0)
            # pengambilan data idf berdasarkan perhitungan sebelumnya
            idf = self.idf[term]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=10):
        if self.documents is None:
            raise ValueError("Model not fitted. Call fit() first.")

        query_terms = query.split()
        scores = []

        for i in range(len(self.documents)):
            score = self._score_document(query_terms, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [(idx, score) for idx, score in scores if score > 0][:top_k]

    def save(self, path):
        """Simpan model BM25 ke file .pkl"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model BM25 disimpan ke: {path}")

    def load(path):
        """Muat kembali model BM25 dari file .pkl"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model BM25 dimuat dari: {path}")
        return model