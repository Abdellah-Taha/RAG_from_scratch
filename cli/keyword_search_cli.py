import argparse
import json
import os
import pickle
import re
import string
import sys
from typing import Counter

from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies: dict[int, Counter] = {} 
        self.stemer = PorterStemmer()

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            stemmed_token = self.stemer.stem(token)
            self.index.setdefault(stemmed_token, set()).add(doc_id)
            self.term_frequencies[doc_id] += 1

    def get_documents(self, term):
        return sorted(self.index.get(self.stemer.stem(term), set()))

    def get_tf(self, doc_id, term):
        try:
            return self.term_frequencies[doc_id][term]
        except BaseException:
            return 0
    
    def tokenizer(self, term):
        tokenized_term = tokenize_text(term)
        
    
    
    def build(self, path):
        movies = extract_data(path)
        for movie in movies["movies"]:
            self.docmap[movie["id"]] = movie
            text = movie["description"] + movie["title"]
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
            
def remove_stop_words(tokens, stop_words, stemer):
    res = []
    for token in tokens:
        token = stemer.stem(token)
        if token not in stop_words:
            res.append(token)
    return res

def tokenize_text(token):
    clean_token = ''.join(ch for ch in token.lower() if ch not in string.punctuation)
    clean_token = re.split(" |\t|\r|\r|\f", clean_token)
    return clean_token

def get_clean_tokens(token, stop_words, stemer):
    """returns a list of strings of a clean stemed string
        removing all the punctuations and spaces
    """
    clean_token = tokenize_text(token)
    clean_token = remove_stop_words(clean_token, stop_words, stemer)
    return clean_token

    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build dataset")

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
            for token in tokenize_text(args.query):
                matching_documents.update(indexer.get_documents(token))

            for doc_id in sorted(matching_documents)[:5]:
                print(f"{indexer.docmap[doc_id]['title']} ({doc_id})")
        case "build":
            indexer = InvertedIndex()
            indexer.build("data/movies.json")
            indexer.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()