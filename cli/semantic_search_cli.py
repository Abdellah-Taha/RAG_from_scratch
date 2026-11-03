import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="commands",
                                        help="display available commands")
    sub_parser.add_parser("verify",
                          help="display model details")

    embed_parser = sub_parser.add_parser("embed_text", help="embedd a text given as arg")
    embed_parser.add_argument("text", type=str)
    
    sub_parser.add_parser("verify_embeddings")
    
    args = parser.parse_args()

    match args.commands:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()