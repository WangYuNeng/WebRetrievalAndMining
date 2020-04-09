from model import Model
from query import Query

if __name__ == "__main__":
    model = Model("../dataset/model/inverted-file", "../dataset/model/vocab.all", "../dataset/model/file-list")
    queries = Query.from_xml("../dataset/queries/query-train.xml")
    output_name = list()
    for i, q in enumerate(queries):
        print("processing query {}".format(i))
        q.to_term_vec(model.vocab_to_idx, model.term_to_tidx)
        doc_names = model.retrieve(q)
        output_name.append(doc_names)

    with open("out.xml", "w") as f:
        f.write("query_id,retrieved_docs")
        for i, row in enumerate(output_name):
            f.write("\n{},".format(str(i+1).zfill(3)))
            for i, doc in enumerate(row):
                if i == 0:
                    f.write(doc)
                else:
                    f.write(" {}".format(doc))

    
