"""Exercise the actual extractor function without loading Qwen weights."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import BatchEncoding

from backend.model.context_parser import parse_context
from backend.schemas.context import UserContext


def extractor_with_fakes(text):
    # Importing qwen_extractor initializes a large model, so compile its actual
    # function body with injected model/tokenizer dependencies instead.
    source = Path(__file__).resolve().parents[1] / 'model/qwen_extractor.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'extract_context')
    inputs = BatchEncoding({'input_ids': torch.tensor([[10, 11, 12]]),
                            'attention_mask': torch.tensor([[1, 1, 1]])})
    tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=inputs), eos_token_id=0,
                                decode=Mock(return_value=json.dumps(text)))
    model = SimpleNamespace(device='cpu', generate=Mock(return_value=torch.tensor([[10, 11, 12, 20, 21]])))
    namespace = dict(json=json, torch=torch, tokenizer=tokenizer, model=model,
                     UserContext=UserContext, parse_context=parse_context, SYSTEM_PROMPT='Extract context.')
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['extract_context'], tokenizer, model


def test_batch_encoding_inputs_generate_and_slice_correctly():
    extract, tokenizer, model = extractor_with_fakes({'emotions': ['anxiety'], 'intensity': 0.5, 'readiness': 'medium'})
    result = extract('I feel anxious.')
    assert result.emotions == ['anxiety']
    assert tokenizer.apply_chat_template.call_args.kwargs['return_dict'] is True
    assert not model.generate.call_args.args
    assert model.generate.call_args.kwargs['input_ids'].shape == (1, 3)
    assert 'attention_mask' in model.generate.call_args.kwargs
    assert tokenizer.decode.call_args.args[0].tolist() == [20, 21]


def test_unsubstantiated_consent_remains_false_after_extraction_fix():
    extract, _, _ = extractor_with_fakes({'readiness': 'high', 'intensity': 0.3,
                                          'consent_to_exercise': True, 'evidence': ['I consent']})
    assert extract('I feel worried.').consent_to_exercise is False
