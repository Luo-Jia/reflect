import json
from types import SimpleNamespace
from unittest.mock import Mock

from backend.schemas.context import UserContext
from backend.workflow.prompt_workflow import answer_prompt


def make_backend(articles):
    return SimpleNamespace(extract=Mock(return_value=UserContext(needs_clarification=True)),
                           search=Mock(return_value=articles),
                           generate=Mock(return_value='Here are the retrieved articles.'))


def test_paper_request_searches_without_planning_context_and_preserves_links():
    backend = make_backend([{'title': 'A supplied paper', 'abstract': 'A supplied abstract',
                            'url': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pmid': '123', 'doi': '10.example/test'}])
    result = answer_prompt('Can you share research articles about sleep?', backend=backend)
    backend.search.assert_called_once()
    assert backend.search.call_args.args[0] == 'cognitive behavioral therapy insomnia'
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['research_access'] == {'requested': True, 'available_count': 1, 'status': 'retrieved'}
    assert payload['research'][0]['url'] == 'https://pubmed.ncbi.nlm.nih.gov/123/'
    assert payload['research'][0]['doi'] == '10.example/test'
    assert result['plan']['selected_path'] == []
    assert not result['context']['consent_to_exercise']


def test_empty_index_is_reported_as_empty_not_unavailable():
    backend = make_backend([])
    answer_prompt('Please share some research papers.', backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['research_access']['status'] == 'empty'


def test_search_failure_is_distinguished_from_empty_index():
    backend = make_backend([])
    backend.search.side_effect = RuntimeError('Database offline')
    result = answer_prompt('Please share some research papers.', backend=backend)
    payload = json.loads(backend.generate.call_args.args[0][-1]['content'])
    assert payload['research_access']['status'] == 'unavailable'
    assert result['warnings']


def test_explicit_research_request_does_not_forward_private_details():
    backend = make_backend([])
    answer_prompt('I work at PRIVATE_COMPANY. Share research articles about anxiety.', backend=backend)
    query = backend.search.call_args.args[0]
    assert query == 'cognitive behavioral therapy anxiety'
    assert 'PRIVATE_COMPANY' not in query


def test_ordinary_clarification_does_not_trigger_research():
    backend = make_backend([])
    answer_prompt('Hello', backend=backend)
    backend.search.assert_not_called()
