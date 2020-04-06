import numpy as np
import xml.etree.ElementTree as ET

QUERY_STOP_WORDS = {"，", "、", "。"}

class Query:

    def __init__(self, attrib):
        self.attrib_ = attrib
        self.term_vec = None
        pass

    def to_term_vec(self, vocab_to_idx, term_to_idx):
        self.term_vec = np.zeros(len(vocab_to_idx), dtype=int)
        for key in self.attrib_:
            if key == 'number':
                continue
            text = self.attrib_[key]

            # consider chinese word, bigram only
            prev_word = None
            for word in text:
                if word in QUERY_STOP_WORDS:
                    prev_word = None
                    continue
                if prev_word != None:
                    bigram = prev_word + word
                    if bigram in vocab_to_idx:
                        self.term_vec[vocab_to_idx[bigram]] += 1
                if word in vocab_to_idx:
                    self.term_vec[vocab_to_idx[word]] += 1
                prev_word = word

    @staticmethod
    def from_xml(file_name):
        queries = list()
        tree = ET.parse(file_name)
        root = tree.getroot()
        for query in root:
            attribs = dict()
            for attribute in query:
                attribs[attribute.tag] = attribute.text
            attribs['number'] = int(attribs['number'][-3:]) # keep id only
            attribs['question'] = attribs['question'][3:-1] # remove \n and "查詢"
            attribs['narrative'] = attribs['narrative'][5:-1] # remove \n and "相關文件內容"
            attribs['concepts'] = attribs['concepts'][1:-1] # remove \n
            queries.append(Query(attribs))
        return queries

# testing
if __name__ == "__main__":
    qs = Query.from_xml("../dataset/queries/query-train.xml")
    for q in qs:
        print(q.attrib_)
