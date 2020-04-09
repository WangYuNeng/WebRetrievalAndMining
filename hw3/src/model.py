import numpy as np
import time
from math import log, sqrt

NUM_TERMS = 1193467
class Model:

    def __init__(self, inverted_file, vocab_file, doc_file_list):
        self.vocab_to_idx, self.idx_to_vocab = self.build_vocab(vocab_file)
        self.doc_vecs, self.doc_lengths, self.doc_names, self.term_to_tidx, self.tidx_to_docfreq = self.build_td_matrix(inverted_file, doc_file_list)
        pass
    
    def retrieve(self, query):
        scores = np.array(self.BM25(query))
        doc_rank = np.flip(np.argsort(scores))
        top_k = self._cluster(doc_rank[:100], scores)
        return [self.doc_names[idx] for idx in top_k]


    def BM25(self, query, k1=1.5, b=0.75, k3=20):
        N = len(self.doc_vecs)
        avdl = np.average(self.doc_lengths)
        scores = [0] * N
        for d_idx, d_vec in enumerate(self.doc_vecs):
            dot_product = 0
            square_query = 0
            square_doc = 0
            for t_idx in query.term_vec:
                if t_idx in d_vec: 
                    df = self.tidx_to_docfreq[t_idx]
                    tf = d_vec[t_idx]
                    qtf = query.term_vec[t_idx]
                    dl = self.doc_lengths[d_idx]
                    IDF = log((N - df +0.5) / (df + 0.5))
                    TF = (k1 + 1) * tf / (k1 * (1 - b + b * dl / avdl) + tf) 
                    QTF = (k3 + 1) * tf / (k3 + qtf)
                    square_doc += (IDF * TF) ** 2
                    square_query += QTF ** 2
                    dot_product += IDF * TF * QTF
            if square_query != 0 and square_doc != 0:
                scores[d_idx] = dot_product / sqrt(square_query * square_doc)
        return scores

    def _cluster(self, index, score):
        max_iter = 100
        c1 = score[index[len(index) // 4]]
        c2 = score[index[len(index) // 2]]
        for it in range(max_iter):
            sep_i = 0

            for i, doc_idx in enumerate(index):
                val = score[doc_idx]
                if abs(c1 - val) > abs(val - c2):
                    sep_i = i
                    break
            new_c1 = np.average(score[index[:sep_i]])
            new_c2 = np.average(score[index[sep_i:]])
            if (new_c1 == c1 and new_c2 == c2):
                break
            c1 = new_c1
            c2 = new_c2
        return index[:sep_i]

    @staticmethod
    def build_vocab(file_name):
        vocab_to_idx = dict()
        idx_to_vocab = list()
        with open(file_name, "r", encoding='utf8') as f:
            for idx, vocab in enumerate(f.readlines()):
                vocab_to_idx[vocab[:-1]] = idx
                idx_to_vocab.append(vocab)
        return vocab_to_idx, idx_to_vocab

    @staticmethod
    def build_td_matrix(inverted_file, doc_file_list):
        start = time.time()
        with open(doc_file_list, "r", encoding='utf8') as f:
            doc_names = [line.strip("\n").split('/')[-1].lower() for line in f.readlines()]
            num_files = len(doc_names)
        doc_vecs = [{} for _ in range(num_files)]
        doc_lengths = [0] * num_files
        term_to_tidx = dict()
        tidx_to_docfreq = list()
        with open(inverted_file, "r", encoding='utf8') as f:
            lines = f.readlines()
            i = 0
            term_idx = 0
            while(i < len(lines)):
                [vocab_idx1, vocab_idx2, docfreq] = lines[i][:-1].split()
                (vocab_idx1, vocab_idx2, docfreq) = (int(vocab_idx1), int(vocab_idx2), int(docfreq))
                term_to_tidx[vocab_idx1, vocab_idx2] = term_idx
                tidx_to_docfreq.append(int(docfreq))
                for j in range(docfreq):
                    i += 1 
                    line = lines[i]
                    [doc_id, term_freq] = line[:-1].split()
                    doc_vecs[int(doc_id)][term_idx] = int(term_freq)
                    doc_lengths[int(doc_id)] += int(term_freq)
                if(term_idx % 1000 == 0):
                    print("processing {} terms in {} sec".format(term_idx, time.time()-start))
                i += 1
                term_idx += 1
        # print(num_terms)
        return doc_vecs, doc_lengths, doc_names, term_to_tidx, tidx_to_docfreq
                



# testing
if __name__ == "__main__":
    model = Model(" ", "../dataset/model/vocab.all", " ")
    print(model.vocab_to_idx["Copper"])
    print(model.vocab_to_idx["EGCG"])
