import argparse
import json
import string
import re
from nltk.stem import PorterStemmer


def extract_data(path: str):
    with open(path, "r") as f:
        text = f.read()
    return json.loads(text)

def get_stop_words(path):
    stop_words = []
    with open(path,"r") as f:
        for line in f:
            stop_words.append(line.strip("\n"))
    return stop_words
            
def get_clean_tokens(token, stop_words, stemer):
    clean_token = ''.join(ch for ch in token.lower() if ch not in string.punctuation)
    clean_token = re.split(" |\t|\r|\r|\f", clean_token)
    clean_token = remove_stop_words(clean_token, stop_words, stemer)
    return clean_token

def remove_stop_words(tokens, stop_words, stemer):
    res = []
    for token in tokens:
        token = stemer.stem(token)
        if token not in stop_words:
            res.append(token)
    return res
    
def main() -> None:
    stemer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    data = extract_data("data/movies.json")
    res = []
    match args.command:
        case "search":
            stop_words = get_stop_words("data/stop_words.txt")
            for movie in data["movies"]:
                clean_title = get_clean_tokens(movie["title"], stop_words, stemer)
                query = get_clean_tokens(args.query, stop_words, stemer)
                match = any(word1 in word2 for word1 in query for word2 in clean_title) 
                if match:
                    res.append(movie["title"])
            print(f"Searching for: {args.query}")
            for i, name in enumerate(res):
                print(f"{i}. {name}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()