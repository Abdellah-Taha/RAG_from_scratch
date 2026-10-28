import argparse
import json
import os
import pickle
import re
import string
import sys
import math
from typing import Counter

from nltk.stem import PorterStemmer

BM25_K1 = 1.5

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies: dict[int, Counter] = {} 
        self.stemer = PorterStemmer()

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text, self.stemer)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        return sorted(self.index.get(self.stemer.stem(term), set()))

    def get_tf(self, doc_id, term):
        try:
            tf = int(self.term_frequencies[doc_id][term])
            return tf
        except BaseException:
            return 0
    
    def get_idf(self, term):
        token = self.tokenize_single_term(term)
        document_frequency = len(self.index.get(token, set()))
        idf = math.log((len(self.docmap) + 1) / (document_frequency + 1))
        return idf
    
    def tokenize_single_term(self, term):
        tokenized_term = tokenize_text(term, self.stemer)
        if not tokenized_term or not tokenized_term[0]:
            raise ValueError("No searchable term found")
        if len(tokenized_term) > 1:
            raise Exception("Detected more than 1 term")
        return tokenized_term[0]
    
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, set()))
        bm25_score = math.log((N - df + .5) / (df + .5) + 1)
        return bm25_score
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        tf_saturation = (tf * (k1 + 1)) / (tf + k1)
        return tf_saturation
    
    def build(self, path):
        movies = extract_data(path)
        for movie in movies["movies"]:
            self.docmap[movie["id"]] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], text)

    def save(self):
        path = "cache"
        os.makedirs (path, exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)

def extract_data(path: str):
    with open(path, "r") as f:
        text = f.read()
    return json.loads(text)

def get_stop_words(path):
    """returns a list of the stop words"""
    stop_words = []
    with open(path,"r") as f:
        for line in f:
            stop_words.append(line.strip("\n"))
    return stop_words
            
def remove_stop_words(tokens, stop_words):
    res = []
    for token in tokens:
        if token not in stop_words:
            res.append(token)
    return res

def tokenize_text(token, stemer=None):
    if stemer is None:
        stemer = PorterStemmer()
    clean_token = ''.join(ch for ch in token.lower() if ch not in string.punctuation)
    clean_token = re.split(" |\t|\r|\n|\f", clean_token)
    for i in range(len(clean_token)):
        clean_token[i] = stemer.stem(clean_token[i])
    return clean_token

def get_clean_tokens(token, stop_words, stemer):
    """returns a list of strings of a clean stemed string
        removing all the punctuations and spaces
    """
    clean_token = tokenize_text(token, stemer)
    clean_token = remove_stop_words(clean_token, stop_words)
    return clean_token

def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = inverted_index.tokenize_single_term(term)
    score: float = inverted_index.get_bm25_idf(token)
    return score
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    
    subparsers.add_parser("build", help="Build dataset")
    
    tf_parser = subparsers.add_parser("tf", help="print's the term frequency for a term in a given document ID")
    tf_parser.add_argument("Document_ID", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="the term to search for (must be a SINGLE token!)")
    
    idf_parser = subparsers.add_parser("idf", help="print to stdout the Inversed Term Frequency")
    idf_parser.add_argument("term", type=str, help="the term to search for (must be a SINGLE token!)")
    
    tf_idf_parser = subparsers.add_parser("tfidf", help="print to stdout the final TF-IDF score of a term")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="The term to calculate")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf",
                                            help="Get BM25 IDF score for a given term"
                                            )
    bm25_idf_parser.add_argument("term",
                                 type=str,
                                 help="Term to get BM25 IDF score for"
                                 )

    bm25_tf_parser = subparsers.add_parser("bm25tf",
                                           help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id",
                                type=int,
                                help="Document ID")
    bm25_tf_parser.add_argument("term",
                                type=str,
                                help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1",
                                type=float,
                                nargs="?",
                                default=BM25_K1,
                                help="Tunable BM25 K1 parameter")

    args = parser.parse_args()
    # data = extract_data("data/movies.json")
    # res = []
    match args.command:
        case "search":
            indexer = InvertedIndex()
            try:
                indexer.load()
            except FileNotFoundError as e:
                print(f"Error loading index: {e}", file=sys.stderr)
                sys.exit(1)

            matching_documents = set()
            for token in tokenize_text(args.query, indexer.stemer):
                matching_documents.update(indexer.get_documents(token))

            for doc_id in sorted(matching_documents)[:5]:
                print(f"{indexer.docmap[doc_id]['title']} ({doc_id})")
        case "build":
            indexer = InvertedIndex()
            indexer.build("data/movies.json")
            indexer.save()
        case "tf":
            try:
                inverted_indexer = InvertedIndex()
                inverted_indexer.load()
                token = inverted_indexer.tokenize_single_term(args.term)
                print(inverted_indexer.get_tf(args.Document_ID, token))
            except Exception as e:
                print(e)
                sys.exit()
        case "idf":
            try:
                inverted_indexer = InvertedIndex()
                inverted_indexer.load()
                idf = inverted_indexer.get_idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except Exception as e:
                print(e)
        case "tfidf":
            inverted_indexer = InvertedIndex()
            inverted_indexer.load()
            token = inverted_indexer.tokenize_single_term(args.term)
            tf = inverted_indexer.get_tf(args.doc_id, token)
            idf = inverted_indexer.get_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            inverted_indexer = InvertedIndex()
            inverted_indexer.load()
            token = inverted_indexer.tokenize_single_term(args.term)
            bm25tf = inverted_indexer.get_bm25_tf(args.doc_id, token)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()