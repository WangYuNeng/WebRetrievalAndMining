import numpy as np
import time
from math import log, sqrt

NUM_TERMS = 1193467
class Model:

    def __init__(self, inverted_file, vocab_file, doc_file_list):
        self.vocab_to_idx, self.idx_to_vocab = self.build_vocab(vocab_file)
        self.doc_vecs, self.doc_lengths, self.doc_names, self.term_to_tidx, self.tidx_to_docfreq, self.tidx_to_term = self.build_td_matrix(inverted_file, doc_file_list)
        pass
    
    def retrieve(self, query):
        start = time.time()
        scores = np.array(self.BM25(query))
        print("BM25(1): {} secs".format(time.time()-start))
        start = time.time()
        doc_rank = np.flip(np.argsort(scores))
        # pseudo_relevance = self._cluster(doc_rank[:30], scores)
        query = self.feedback(query, doc_rank[:10])
        print("Feedback: {} secs".format(time.time()-start))
        start = time.time()
        scores = np.array(self.BM25(query))
        print("BM25(2): {} secs".format(time.time()-start))
        doc_rank = np.flip(np.argsort(scores))
        return [self.doc_names[idx] for idx in doc_rank[:100]]


    def BM25(self, query, k1=1.5, b=0.75, k3=100):
        N = len(self.doc_vecs)
        avdl = np.average(self.doc_lengths)
        scores = [0] * N
        q_vec = query.term_vec
        for d_idx, d_vec in enumerate(self.doc_vecs):
            if len(d_vec) > len(q_vec):
                for t_idx in q_vec:
                    if t_idx in d_vec: 
                        df = self.tidx_to_docfreq[t_idx]
                        tf = d_vec[t_idx]
                        qtf = q_vec[t_idx]
                        dl = self.doc_lengths[d_idx]
                        IDF = log((N - df +0.5) / (df + 0.5))
                        TF = (k1 + 1) * tf / (k1 * (1 - b + b * dl / avdl) + tf) 
                        QTF = (k3 + 1) * qtf / (k3 + qtf)
                        scores[d_idx] += IDF * TF * QTF
            else:
                for t_idx in d_vec:
                    if t_idx in q_vec: 
                        df = self.tidx_to_docfreq[t_idx]
                        tf = d_vec[t_idx]
                        qtf = q_vec[t_idx]
                        dl = self.doc_lengths[d_idx]
                        IDF = log((N - df +0.5) / (df + 0.5))
                        TF = (k1 + 1) * tf / (k1 * (1 - b + b * dl / avdl) + tf) 
                        QTF = (k3 + 1) * qtf / (k3 + qtf)
                        scores[d_idx] += IDF * TF * QTF
        return scores

    def feedback(self, query, relevant_docs, alpha=1, beta=1):
        query.term_vec = {key: alpha * query.term_vec[key] for key in query.term_vec}
        for doc_id in relevant_docs:
            doc_vec = self.doc_vecs[doc_id]
            for term in doc_vec:
                if term in query.term_vec:
                    query.term_vec[term] += beta / len(relevant_docs) * doc_vec[term]
                else:
                    query.term_vec[term] = beta / len(relevant_docs) * doc_vec[term]
        return query

    def _cluster(self, index, score):
        max_iter = 100
        c1 = score[index[len(index) // 3]]
        c2 = score[index[len(index) // 3 * 2]]
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
                idx_to_vocab.append(vocab[:-1])
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
        tidx_to_term = list()
        with open(inverted_file, "r", encoding='utf8') as f:
            lines = f.readlines()
            i = 0
            term_idx = 0
            while(i < len(lines)):
                [vocab_idx1, vocab_idx2, docfreq] = lines[i][:-1].split()
                (vocab_idx1, vocab_idx2, docfreq) = (int(vocab_idx1), int(vocab_idx2), int(docfreq))
                term_to_tidx[vocab_idx1, vocab_idx2] = term_idx
                tidx_to_docfreq.append(int(docfreq))
                tidx_to_term.append((vocab_idx1, vocab_idx2))
                for j in range(docfreq):
                    i += 1 
                    line = lines[i]
                    [doc_id, term_freq] = line[:-1].split()
                    doc_vecs[int(doc_id)][term_idx] = int(term_freq)
                    doc_lengths[int(doc_id)] += int(term_freq)
                if(term_idx % 100000 == 0):
                    print("processing {} terms in {} sec".format(term_idx, time.time()-start))
                i += 1
                term_idx += 1
        # print(num_terms)
        return doc_vecs, doc_lengths, doc_names, term_to_tidx, tidx_to_docfreq, tidx_to_term
                



# testing
if __name__ == "__main__":
    model = Model(" ", "../dataset/model/vocab.all", " ")
    print(model.vocab_to_idx["Copper"])
    print(model.vocab_to_idx["EGCG"])
