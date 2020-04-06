import numpy as np



class Model:

    def __init__(self, inverted_file, vocab_file, doc_file_list):
        self.vocab_to_idx, self.idx_to_vocab = self.build_vocab(vocab_file)
        self.term_doc_matrix, self.term_to_idx, self.idx_to_docfreq = self.build_td_matrix(inverted_file, doc_file_list)
        pass

    @staticmethod
    def build_vocab(file_name):
        vocab_to_idx = dict()
        idx_to_vocab = list()
        with open(file_name, "r") as f:
            for idx, vocab in enumerate(f.readlines()):
                vocab_to_idx[vocab[:-1]] = idx
                idx_to_vocab.append(vocab)
        return vocab_to_idx, idx_to_vocab

    @staticmethod
    def build_td_matrix(inverted_file, doc_file_list):
        with open(doc_file_list, "r") as f:
            num_files = len(f.readlines())
        td_matrix = list()
        term_to_idx = dict()
        idx_to_docfreq = list()
        with open(inverted_file, "r") as f:
            lines = f.readlines()
            i = 0
            while(lines[i] != ""): # remove tailing empty line
                [vocab_idx1, vocab_idx2, docfreq] = lines[i][:-1].split()
                (vocab_idx1, vocab_idx2, docfreq) = (int(vocab_idx1), int(vocab_idx2), int(docfreq))
                term_to_idx[(vocab_idx1, vocab_idx2)] = len(idx_to_docfreq)
                idx_to_docfreq.append(int(docfreq))
                doc_row = [0] * num_files
                for j in range(docfreq):
                    i += 1 
                    line = lines[i]
                    [doc_id, term_freq] = line[:-1].split()
                    doc_row[int(doc_id)] = int(term_freq)
                i += 1
                td_matrix.append(doc_row)

        return np.array(td_matrix), term_to_idx, idx_to_docfreq
                



# testing
if __name__ == "__main__":
    model = Model(" ", "../dataset/model/vocab.all", " ")
    print(model.vocab_to_idx["Copper"])
    print(model.vocab_to_idx["EGCG"])
