import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from backend.schemas.context import UserContext
from backend.model.context_parser import parse_context

MODEL_ID = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
model.eval()

SYSTEM_PROMPT = """
Extract structured context for a CBT-informed wellness agent.

Return ONLY a valid JSON object with:
- emotions: list containing only anxiety, sadness, shame, frustration, anger, neutral; use [] if unknown
- intensity: number between 0 and 1, or null
- readiness: low, medium, high, or unknown
- preference: validation, reflection, action, or unspecified
- automatic_thought: string or null
- thought_patterns: list of tentative user-supported thinking patterns; [] if unknown
- lifestyle_context: list of explicitly stated lifestyle details; [] if unknown
- consent_to_exercise: boolean
- evidence: list of short quotes from the user
- confidence: number between 0 and 1
- needs_clarification: boolean

Rules:
Do not diagnose.
Do not invent emotions, preferences, or consent.
Only mark exercise consent true when explicit.
Use null or unknown for unsupported values.
Use null only for intensity and automatic_thought. Use [] for absent evidence,
0 for unknown confidence, and "unknown" for unsupported readiness.
Do not provide therapy or crisis assessments.
"""
SYSTEM_PROMPT += "\nJSON schema (authoritative field constraints):\n" + json.dumps(
    UserContext.model_json_schema()
)
SYSTEM_PROMPT += "\nExample when information is unknown:\n" + json.dumps(
    UserContext(needs_clarification=True).model_dump()
)

def extract_context(message: str, history=None) -> UserContext:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\nUse prior turns to understand short replies. Extract the current user's context, not the assistant's feelings. Consent must be supported by a quote from the current user message; prior consent does not override a refusal."},
        {"role": "user", "content": json.dumps({
            "conversation": (history or [])[-12:],
            "current_user_message": message,
            "task": "Extract context as JSON for the current user message. Do not continue the conversation."
        })}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=650,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    context = parse_context(text)

    # Fail closed on consent unless explicit evidence
    # appears in the user's message.
    if context.consent_to_exercise:
        if not context.evidence or not all(
            quote.strip() and quote in message for quote in context.evidence
        ):
            context.consent_to_exercise = False

    return context
