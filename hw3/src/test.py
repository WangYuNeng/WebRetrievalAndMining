import sys

if __name__ == "__main__":
    true_file = sys.argv[1]
    test_file = sys.argv[2]

    true_docs = []
    test_docs = []

    with open(true_file, "r") as f:
        lines = f.readlines()
        true_docs = [line.split(",")[1].split() for line in lines[1:]]
    with open(test_file, "r") as f:
        lines = f.readlines()
        test_docs = [line.split(",")[1].split() for line in lines[1:]]
    
    MAP = 0
    for true, retrieved in zip(true_docs, test_docs):
        AP = 0
        precision = 0
        recall = 0
        true_pos = 0
        true_docs = {doc for doc in true}
        for i, doc in enumerate(retrieved):
            if doc in true_docs:
                true_pos += 1
            precision = true_pos / (i+1)
            d_recall = true_pos / len(true) - recall
            recall = true_pos / len(true)
            AP += precision * d_recall
        MAP += AP / len(true_docs)
    print("MAP = ", MAP)