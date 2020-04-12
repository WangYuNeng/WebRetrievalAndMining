from model import Model
from query import Query
import time
import threading
import numpy as np

NUM_THREAD = 4
lock = threading.Lock()
batch_size = 0

def _retrieve_thread(qs, model, batch_id, output_name):
    for i, q in enumerate(qs):
        total_start = time.time
        lock.acquire()
        print("Thread {}: Retrieving query{}".format(batch_id, q.attrib_["number"]))
        lock.release()
        q.to_term_vec(model.vocab_to_idx, model.term_to_tidx)
        
        lock.acquire()
        print("Thread {}: Calculating BM25(1) ...".format(batch_id))
        lock.release()
        start = time.time()
        scores = np.array(model.BM25(q))
        doc_rank = np.flip(np.argsort(scores))
        lock.acquire()
        print("Thread {}: Calculating BM25(1) using {} secs.".format(batch_id, time.time() - start))
        print("Thread {}: Calculating feedback ...".format(batch_id))
        lock.release()
        start = time.time()
        model.feedback(q, doc_rank[:10])
        lock.acquire()
        print("Thread {}: Calculating feedback using {} secs.".format(batch_id, time.time() - start))
        print("Thread {}: Calculating BM25(2) ...".format(batch_id))
        lock.release()
        start = time.time()
        scores = np.array(model.BM25(q))
        doc_rank = np.flip(np.argsort(scores))
        doc_names = [model.doc_names[idx] for idx in doc_rank[:100]]
        print("Thread {}: Calculating BM25(2) using {} secs.".format(batch_id, time.time() - start))

        lock.acquire()
        output_name[int(batch_id*batch_size)+i] = doc_names
        print("Retrieving query{} takes {} secs.".format(int(batch_id*batch_size)+i, time.time() - total_start))
        print("Retrieved docs:")
        print(doc_names)
        lock.release()

if __name__ == "__main__":
    model = Model("../dataset/model/inverted-file", "../dataset/model/vocab.all", "../dataset/model/file-list")
    queries = Query.from_xml("../dataset/queries/query-train.xml")
    output_name = [None for _ in queries]
    start = time.time()
    # NUM_THREAD = len(queries)
    threads = list()
    batch_size = len(queries) / NUM_THREAD
    for batch_id in range(NUM_THREAD - 1):
        threads.append(threading.Thread(target=_retrieve_thread, args=[queries[int(batch_id*batch_size):int((batch_id+1)*batch_size)], model, batch_id, output_name]))
    threads.append(threading.Thread(target=_retrieve_thread, args=[queries[int((NUM_THREAD-1)*batch_size):], model, NUM_THREAD-1, output_name]))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open("out.csv", "w") as f:
        f.write("query_id,retrieved_docs")
        for i, row in enumerate(output_name):
            f.write("\n{},".format(str(queries[i].attrib_["number"]).zfill(3)))
            for i, doc in enumerate(row):
                if i == 0:
                    f.write(doc)
                else:
                    f.write(" {}".format(doc))
    print("Total time usage: {} secs".format(time.time() - start))

    
