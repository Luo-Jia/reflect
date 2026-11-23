from collections import Counter
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DATASET = Path(__file__).parent / "data" / "cbt_tot_synthetic_200.json"
logger = logging.getLogger("backend.main")


def configure_model_logging():
    log_path = Path(__file__).parent / "logs" / "model.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("backend")
    if not any(
        isinstance(handler, RotatingFileHandler)
        and handler.baseFilename == str(log_path.resolve())
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path.resolve(), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def evaluate_dataset():
    from backend.database.loader import load_dataset
    from backend.workflow.workflow import graph

    examples = load_dataset(str(DATASET))

    print(f"Loaded {len(examples)} examples")

    results = []
    for example in examples:
        result = graph.invoke({
            "example": example,
            "user_message": example.user_message,
        })
        results.append(result)

    counts = Counter(
        r.get("selected_intervention") or "safety_bypass"
        for r in results
    )

    print("\nIntervention distribution:")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")

    matches = sum(
        result.get("selected_intervention")
        == example.tot.selected_intervention
        for result, example in zip(results, examples)
    )

    print(f"\nReference agreement: {matches}/{len(examples)}")


def main():
    parser = argparse.ArgumentParser(description="Get a CBT-informed response or search indexed research.")
    parser.add_argument("--prompt", help="Your message; prompts interactively if omitted.")
    parser.add_argument("--search-only", action="store_true", help="Return research articles without generating a response.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of articles to return.")
    parser.add_argument("--evaluate", action="store_true", help="Run the synthetic dataset workflow.")
    parser.add_argument("--web", action="store_true", help="Serve the local browser chatbot.")
    parser.add_argument("--port", type=int, default=8001, help="Local webchat port (default: 8001).")
    args = parser.parse_args()
    configure_model_logging()

    if args.web:
        if args.evaluate or args.search_only or args.prompt is not None:
            parser.error("--web cannot be combined with --evaluate, --search-only, or --prompt")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        import uvicorn
        from backend.web.api import create_app
        print(f"Webchat will be available at http://127.0.0.1:{args.port} after initialization.", flush=True)
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)
        return

    if args.evaluate:
        if args.prompt is not None or args.search_only:
            parser.error("--prompt and --search-only cannot be combined with --evaluate")
        evaluate_dataset()
        return

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if not args.search_only:
        chat(initial_prompt=args.prompt, top_k=args.top_k)
        return

    try:
        prompt = (args.prompt if args.prompt is not None else input("Your message: ")).strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not prompt:
        parser.error("Please enter a non-empty research question")

    from backend.retrieval.search_knowledge import search_research

    articles = search_research(prompt, top_k=args.top_k, fetch_k=max(30, args.top_k))
    if not articles:
        print("No matching articles found in the local research database.")
        return

    for index, article in enumerate(articles, start=1):
        print(f"\n{index}. {article.get('title', 'Untitled')}")
        print(f"Year: {article.get('year', 'Unknown')} | PMID: {article.get('pmid', 'Unknown')}")
        print(f"Abstract: {article.get('abstract') or 'Unavailable'}")
        print(f"Source: {article.get('url') or 'Unavailable'}")


def initialize_chat():
    """Shared startup for the terminal and browser chat; load models once."""
    configure_model_logging()
    from backend.workflow.prompt_workflow import get_default_backend, get_examples
    from backend.conversation.user_profile import load_profile

    profile = load_profile()

    logger.info("Loading language model")
    backend = get_default_backend()
    logger.info("Loading planning examples")
    get_examples()
    logger.info("Initializing research retrieval")
    try:
        from backend.retrieval.search_knowledge import get_embedding_model
        get_embedding_model()
    except Exception as exc:
        logger.warning("Research initialization unavailable: exception=%s", type(exc).__name__)

    return backend, profile


def chat(initial_prompt=None, top_k=5):
    from backend.conversation.session_state import SessionState
    from backend.workflow.prompt_workflow import answer_prompt

    backend, profile = initialize_chat()

    session = SessionState(profile_id=profile.profile_id)
    from backend.ingestion.ingest_journal import ingest_journal, load_profile_journal
    journal_entries = load_profile_journal(profile.profile_id)
    if journal_entries:
        ingest_journal(session, journal_entries)
    logger.info("Loaded %d journal entries for the active profile", len(journal_entries))
    pending = initial_prompt
    print("Let's talk. Type 'exit' or 'quit' to end the conversation.")
    while True:
        try:
            message = (pending if pending is not None else input("\nYou: ")).strip()
            pending = None
            if message.lower() in {"exit", "quit"}:
                print("Goodbye.")
                return
            if not message:
                continue

            result = answer_prompt(message, top_k=top_k, session=session, backend=backend, profile=profile)
            for warning in result["warnings"]:
                print(f"Notice: {warning}")
            print(f"\nAssistant: {result['response']}")
            if result["articles"]:
                print("\nRetrieved sources:")
                for index, article in enumerate(result["articles"], 1):
                    print(f"[{index}] {article.get('title')} — {article.get('url') or 'URL unavailable'}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return


if __name__ == "__main__":
    main()
