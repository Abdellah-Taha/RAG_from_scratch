import argparse

from lib.semantic_search import verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="commands",
                                        help="display available commands")
    sub_parser.add_parser("verify",
                          help="display model details")
    args = parser.parse_args()

    match args.commands:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()