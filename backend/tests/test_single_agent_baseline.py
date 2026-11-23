from types import SimpleNamespace
from unittest.mock import Mock

from backend.evaluation.single_agent import answer_baseline
from backend.evaluation.evaluate_conversations import run_cases
from backend.schemas.context import UserContext


def test_baseline_uses_one_generation_and_no_extractor_or_critic():
    backend = SimpleNamespace(generate=Mock(return_value='I am here to listen.'),
                              extract=Mock(return_value=UserContext()), delegate=Mock(), search=Mock(return_value=[]))
    result = answer_baseline('I feel anxious about work.', backend=backend)
    assert result['validation']['valid']
    backend.generate.assert_called_once()
    backend.extract.assert_not_called()
    backend.delegate.assert_not_called()


def test_baseline_evaluation_records_architecture_and_time():
    report = run_cases([{'id': 'test', 'turns': [{'message': 'Hello', 'response': 'Hello.'}]}], baseline=True)
    assert report['failed_turns'] == 0
    assert report['architecture'] == 'single_agent_baseline'
    assert report['records'][0]['elapsed_seconds'] >= 0
