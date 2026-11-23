from backend.retrieval.search_knowledge import search_research


def generate_research_context(question, top_k: int = 5, fetch_k=30, lambda_mult=0.7):

    articles = search_research(
        question,
        top_k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    context = []

    for article in articles:

        context.append(f"""
        PMID: {article['pmid']}
        Title: {article['title']}
        Journal: {article['journal']}
        Year: {article['year']}
        Abstract: {article['abstract']}
        Source: {article['url']}
        Publication types: {article['publication_types']}
        Evidence classification: {article['evidence_type']}
        Retrieval score: {article['evidence_score']}
        """)

    return "\n\n".join(context)