"""Conservative rule-based safety routing; not a validated crisis classifier.

Explicit present-tense danger gets an immediate response. Other recognized
safety language gets a safety question, not an automatic emergency label.
Unmatched language is not proof that a person is safe.
"""
import re

SAFETY_QUESTION = "Are you in immediate danger or about to act on these thoughts right now?"
EMERGENCY_RESPONSE = (
    "Your immediate safety comes first. If you are in the United States, "
    "call 911 now; elsewhere, call your local emergency number. "
    "If possible, ask someone nearby to help you call and stay with you. "
    "This chat cannot contact emergency services for you."
)
SAFETY_RESPONSE = (
    "Thank you for telling me. I want to check your immediate safety before "
    "continuing. " + SAFETY_QUESTION + " If so, call 911 in the United States "
    "or your local emergency number elsewhere."
)

_IMMEDIATE = re.compile(
    r"\b(?:i am|i'm) (?:in immediate danger|about to "
    r"(?:hurt myself|injure myself|kill myself|end my life|hurt someone|kill someone))\b"
    r"|\bi (?:am going to|plan to|will) (?:kill myself|end my life|hurt myself|injure myself|"
    r"kill someone|hurt someone) (?:right now|now|tonight|today)\b"
    r"|\bi (?:have |just |have just )?(?:overdosed|taken an overdose)\b"
    r"|\bi (?:cannot|can't) (?:breathe|keep myself safe|stop the bleeding)\b",
    re.IGNORECASE,
)
_CONCERN = re.compile(
    r"\b(?:suicid\w*|self[- ]harm\w*|overdos\w*|hurt myself|injure myself|kill myself|"
    r"end my life|hurt someone|kill someone|immediate danger|"
    r"don't want to live|do not want to live|wish i were dead)\b",
    re.IGNORECASE,
)


def assess_safety(message, history=()):
    text = message.lower().replace('\u2019', "'")
    # A direct answer to the application's safety question uses prior context.
    if text.strip().rstrip('.!') in {'yes', 'yes i am', 'yes right now'}:
        previous = list(history)[-1:]
        if previous and previous[0].get('role') == 'assistant' and SAFETY_QUESTION in previous[0].get('content', ''):
            return 'emergency'

    # Historical, conditional, and quoted language needs clarification rather
    # than treating a quoted or past event as a confirmed current emergency.
    for clause in re.split(r'[.!?;\n]|\bbut\b', text):
        if re.search(r'\b(?:used to|yesterday|last year|years? ago|if|what if|said|wrote|remember)\b|["“”]', clause):
            continue
        if _IMMEDIATE.search(clause):
            return 'emergency'
    if _CONCERN.search(text):
        return 'safety_check'
    return 'standard_cbt'


def safety_response_for(route):
    return EMERGENCY_RESPONSE if route == 'emergency' else SAFETY_RESPONSE
