"""Retrieval tools shared through validated graph state, not independent agents."""
import re
from backend.agents.contracts import EvidenceRecord
from backend.retrieval.example_retriever import similar_examples
from backend.retrieval.journal_retriever import retrieve_journal, asks_about_journal


def asks_about_research(message):
    return bool(re.search(r'\b(?:research|studies|study|papers?|articles?|citations?|sources?|scientific evidence)\b', message, re.I))


def research_query(message, context):
    # Only allowlisted public topics reach retrieval, never raw personal text.
    topics = [label for pattern, label in [
        (r'\banxi(?:ety|ous)\b', 'anxiety'),
        (r'\bdepress(?:ion|ed)\b', 'depression'),
        (r'\b(?:sleep|insomnia)\b', 'insomnia'),
        (r'\bstress\b', 'stress'),
        (r'\b(?:negative thoughts?|cognitive restructuring)\b', 'cognitive restructuring'),
    ] if re.search(pattern, message, re.I)]
    topic = ' '.join(topics) or ' '.join(context.emotions) or 'emotional support'
    return f'cognitive behavioral therapy {topic}'


def retrieve_context(message, session, context, backend, top_k, eligible):
    available = [entry for entry in session.journal_entries if entry.get('profile_id') == session.profile_id]
    journal = retrieve_journal(message, available)
    access = {
        'profile_id': session.profile_id, 'available_count': len(available),
        'retrieved_count': len(journal),
        'catalog': [{key: entry[key] for key in ('entry_id', 'date', 'title', 'synthetic')}
                    for entry in available] if asks_about_journal(message) else [],
    }
    examples = similar_examples(message) if eligible else []
    articles, warnings = [], []
    if eligible or asks_about_research(message):
        try:
            articles = backend.search(research_query(message, context), top_k=top_k, fetch_k=max(30, top_k))[:top_k]
        except Exception as exc:
            warnings.append(f'Research retrieval unavailable: {exc}')
    evidence = [EvidenceRecord(source=index, title=item.get('title'),
                               abstract=(item.get('abstract') or '')[:2500],
                               url=item.get('url'), doi=item.get('doi'), pmid=item.get('pmid')).model_dump()
                for index, item in enumerate(articles, 1)]
    return examples, articles, evidence, journal, access, warnings
