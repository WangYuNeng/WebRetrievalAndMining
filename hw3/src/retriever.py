from model import Model
from query import Query

if __name__ == "__main__":
    model = Model("../dataset/model/inverted-file", "../dataset/model/vocab.all", "../dataset/model/file-list")
    queries = Query.from_xml("../dataset/queries/query-train.xml")
    # for q in queries:
        # q.to_term_vec(model.vocab_to_idx, model.term_to_idx)
        # for idx, freq in enumerate(q.term_vec):
            # if freq != 0:
                # print("{} freq: {}".format(model.idx_to_vocab[idx], freq))
        # input()
    
