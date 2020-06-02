from model import Model
from query import Query
import argparse
import time

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-r', action="store_true")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-d', '--NTCIR_dir', required=True)
    parser.add_argument('-f', type=int, default=1)
    parser.add_argument('-k1', type=float, default=1.5)
    parser.add_argument('-b', type=float, default=0.75)
    parser.add_argument('-k3', type=float, default=50)
    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-beta', type=float, default=0.2)
    parser.add_argument('-concepts_w', type=float, default=5)
    


    args = parser.parse_args()
    print(args.r, args.f, args.k1, args.b, args.k3, args.alpha, args.beta)
    model = Model(args.model_dir + "/inverted-file", args.model_dir + "/vocab.all", args.model_dir + "/file-list")
    queries = Query.from_xml(args.input)
    output_name = list()
    for i, q in enumerate(queries):
        print("processing query {}".format(q.attrib_["number"]))
        q.to_term_vec(model.vocab_to_idx, model.term_to_tidx, args.concepts_w)
        doc_names = model.retrieve(q, args.r, args.f, args.k1, args.b, args.k3, args.alpha, args.beta)
        output_name.append(doc_names)

    with open(args.output, "w") as f:
        f.write("query_id,retrieved_docs")
        for i, row in enumerate(output_name):
            f.write("\n{},".format(str(queries[i].attrib_["number"]).zfill(3)))
            for i, doc in enumerate(row):
                if i == 0:
                    f.write(doc)
                else:
                    f.write(" {}".format(doc))

    
