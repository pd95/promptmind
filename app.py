from app.tracing import tracer
from app.settings import Settings
import argparse

from app.commands.index_command import index_command
from app.commands.query_command import query_command
from app.commands.chat_command import chat_command

def main() -> None:
    settings, unknown_args = Settings.from_env_and_args()
    print(f"Using settings: {settings}")

    parser = argparse.ArgumentParser(description="PromptMind CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Index command
    parser_index = subparsers.add_parser("index", help="Create a vector store index from directories or URLs")
    parser_index.add_argument("sources", nargs="+", help="Directories or URLs to index")

    # Query command
    parser_query = subparsers.add_parser("query", help="Query the knowledge base")
    parser_query.add_argument("prompt", help="Your question")

    args, unknown_args = parser.parse_known_args(unknown_args)

    if args.command == "index":
        index_command(args, settings)
    elif args.command == "query":
        query_command(args, settings)
    else:
        chat_command(settings)

if __name__ == "__main__":
    with tracer.start_as_current_span("main()"):
        main()