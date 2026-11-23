
import os
import time
import requests
import xml.etree.ElementTree as ET

from typing import Any
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__spec__.name if __spec__ else __name__)

load_dotenv()

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

HEADERS = {
    "User-Agent": (
        f"CBTResearchAgent/1.0 ({NCBI_EMAIL})"
        if NCBI_EMAIL
        else "CBTResearchAgent/1.0"
    )
}

_last_request = 0.0


def ncbi_request(endpoint: str, params: dict) -> requests.Response:
    """Make a rate-limited request to NCBI."""

    global _last_request

    # NCBI rate limit: 3 requests/sec without API key.
    # Up to 10 requests/sec with API key.
    min_interval = 0.11 if NCBI_API_KEY else 0.34

    elapsed = time.monotonic() - _last_request

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    request_params = {
        **params,
        "tool": "CBTResearchAgent",
    }

    if NCBI_EMAIL:
        request_params["email"] = NCBI_EMAIL

    if NCBI_API_KEY:
        request_params["api_key"] = NCBI_API_KEY

    response = requests.get(
        f"{BASE_URL}/{endpoint}.fcgi",
        params=request_params,
        headers=HEADERS,
        timeout=30,
    )

    _last_request = time.monotonic()

    response.raise_for_status()

    return response


def extract_text(element: Any) -> str:
    """Extract text from an XML element."""

    if element is None:
        return ""

    return "".join(element.itertext()).strip()


def fetch_articles(pmids: list[str]) -> list[dict]:
    """
    Retrieve PubMed article details using EFetch.

    Returns:
        List of article metadata and abstracts.
    """

    if not pmids:
        return []

    # Validate PMIDs.
    for pmid in pmids:
        if not str(pmid).isdigit():
            raise ValueError(f"Invalid PMID: {pmid}")

    response = ncbi_request(
        "efetch",
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
    )

    root = ET.fromstring(response.content)

    articles = []

    for article in root.findall(".//PubmedArticle"):

        citation = article.find("./MedlineCitation")

        if citation is None:
            continue

        article_data = citation.find("./Article")

        if article_data is None:
            continue

        pmid = extract_text(
            citation.find("./PMID")
        )

        title = extract_text(
            article_data.find("./ArticleTitle")
        )

        # Extract abstract sections.
        abstract_parts = []

        for part in article_data.findall(
            "./Abstract/AbstractText"
        ):
            label = part.attrib.get("Label", "")
            content = extract_text(part)

            if label:
                abstract_parts.append(
                    f"{label}: {content}"
                )
            else:
                abstract_parts.append(content)

        abstract = "\n".join(abstract_parts)

        # Extract authors.
        authors = []

        for author in article_data.findall(
            "./AuthorList/Author"
        ):

            last = extract_text(
                author.find("./LastName")
            )

            first = extract_text(
                author.find("./ForeName")
            )

            name = " ".join(
                x for x in [first, last] if x
            )

            if name:
                authors.append(name)

        # Extract journal.
        journal = extract_text(
            article_data.find("./Journal/Title")
        )

        # Extract publication year.
        year = extract_text(
            article_data.find(
                "./Journal/JournalIssue/PubDate/Year"
            )
        )

        if not year:
            year = extract_text(
                article_data.find(
                    "./Journal/JournalIssue/PubDate/MedlineDate"
                )
            )

        # Extract DOI.
        doi = ""

        for identifier in article.findall(
            ".//ELocationID"
        ):

            if identifier.attrib.get("EIdType") == "doi":
                doi = extract_text(identifier)
                break

        # Extract PubMed publication types
        publication_types = []

        for pub_type in article.findall(
            "./PublicationTypeList/PublicationType"
        ):
            publication_types.append(extract_text(pub_type))

        # Extract MeSH terms
        mesh_terms = []

        for mesh in citation.findall("./MeshHeadingList/MeshHeading"):
            descriptor = extract_text(
                mesh.find("./DescriptorName")
            )

            if descriptor:
                mesh_terms.append(descriptor)

        articles.append({
            "pmid": pmid,
            "title": title,
            "authors": authors[:10],
            "journal": journal,
            "year": year,
            "doi": doi,
            "abstract": abstract,
            "url": (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            ),
            "publication_types": publication_types,
            "mesh_terms": mesh_terms,
        })

    return articles


def search_pubmed(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search PubMed and retrieve article metadata.

    Args:
        query:
            PubMed search query.

        max_results:
            Number of articles to retrieve (1-10).

    Returns:
        JSON string containing article metadata.
    """

    import json

    query = query.strip()

    if not query or len(query) > 400:
        raise ValueError("Invalid PubMed search query.")

    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or not 1 <= max_results <= 10
    ):
        raise ValueError(
            "max_results must be between 1 and 10."
        )

    # Step 1: Search PubMed using ESearch.
    search_response = ncbi_request(
        "esearch",
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        },
    )

    search_data = search_response.json()

    search_result = search_data["esearchresult"]

    pmids = search_result.get("idlist", [])

    if not pmids:
        return json.dumps({
            "query": query,
            "total_found": search_result.get("count", "0"),
            "retrieved": 0,
            "articles": [],
        })

    # Step 2: Retrieve article details using EFetch.
    articles = fetch_articles(pmids)

    # Step 3: Return structured results.
    result = {
        "query": query,
        "total_found": search_result.get("count", "0"),
        "retrieved": len(articles),
        "articles": articles,
    }

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    from backend.main import configure_model_logging
    configure_model_logging()

    query = (
        '("Cognitive Behavioral Therapy"[Title/Abstract]) '
        'AND ("Anxiety"[Title/Abstract])'
    )

    logger.info("Searching PubMed")

    results = search_pubmed(
        query=query,
        max_results=5,
    )

    print(results)
