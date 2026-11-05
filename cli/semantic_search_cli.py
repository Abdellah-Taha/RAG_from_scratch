import argparse
import json

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="commands",
                                        help="display available commands")
    sub_parser.add_parser("verify",
                          help="display model details")

    embed_parser = sub_parser.add_parser("embed_text", help="embedd a text given as arg")
    embed_parser.add_argument("text", type=str)
    
    sub_parser.add_parser("verify_embeddings")
    
    embed_query_parser = sub_parser.add_parser("embed_query")
    embed_query_parser.add_argument("query", type=str)
    
    search_parser = sub_parser.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)
    
    args = parser.parse_args()

    match args.commands:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "rb") as f:
                docs = json.load(f)["movies"]
            semantic_search.load_or_create_embeddings(docs)
            results = semantic_search.search(args.query, args.limit)
            for i, res in enumerate(results):
                print(f"{i}. {res["title"]} (score: {res["score"]})")
                print(res["description"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()