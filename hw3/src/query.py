import numpy as np
import xml.etree.ElementTree as ET

QUERY_STOP_WORDS = {"，", "、", "。"}

class Query:

    def __init__(self, attrib):
        self.attrib_ = attrib
        self.term_vec = None
        pass

    def to_term_vec(self, vocab_to_idx, term_to_idx):
        self.term_vec = dict()
        for key in self.attrib_:
            if key == 'number':
                continue
            text = self.attrib_[key]

            # consider chinese word, bigram only
            prev_idx = None
            for word in text:
                if word in QUERY_STOP_WORDS or not (word in vocab_to_idx):
                    prev_idx = None
                    continue
                idx = vocab_to_idx[word]
                if prev_idx != None:
                    bigram = (prev_idx, idx)
                    if bigram in term_to_idx:
                        if term_to_idx[bigram] in self.term_vec:
                            self.term_vec[term_to_idx[bigram]] += 1
                        else:
                            self.term_vec[term_to_idx[bigram]] = 1
                if (idx, -1) in term_to_idx:
                    if term_to_idx[(idx, -1)] in self.term_vec:
                        self.term_vec[term_to_idx[(idx, -1)]] += 1
                    else:
                        self.term_vec[term_to_idx[(idx, -1)]] = 1
                prev_idx = idx

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
