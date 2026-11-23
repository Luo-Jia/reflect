"""Independent, limited rule checks. This is not a clinical safety certification."""
import re
from backend.agents.contracts import OutputCheck

FALLBACK_RESPONSE = (
    "I'm here to listen. Could you tell me what would feel most helpful right now?"
)


def validate_output(response, context, source_count):
    issues = []
    if not isinstance(response, str) or not response.strip():
        issues.append('empty_response')
        response = ''
    if re.search(r'\byou (?:have|suffer from|are diagnosed with) (?:major depressive disorder|bipolar disorder|schizophrenia|clinical depression|an anxiety disorder)\b', response, re.I):
        issues.append('diagnostic_claim')
    if re.search(r'\b(?:you should|you must|i recommend you) (?:stop|start|double|increase|decrease) (?:taking |your )?(?:medication|medications|antidepressants|dose)\b', response, re.I):
        issues.append('medication_instruction')
    if re.search(r'(?:^|\n)\s*(?:SYSTEM UPDATE|ignore (?:all |the )?(?:previous |prior )?instructions)', response, re.I):
        issues.append('instruction_leak')
    citations = [int(value) for value in re.findall(r'\[(?:P)?(\d+)\]', response)]
    if any(value < 1 or value > source_count for value in citations):
        issues.append('unavailable_citation')
    if not context.consent_to_exercise and re.search(
        r"\b(?:let'?s|we will) (?:start|begin|do|complete|try) (?:a |an |the )?(?:CBT |thought[- ]record |restructuring )?exercise\b"
        r'|\b(?:write|list) (?:down )?(?:three |five |your )?(?:automatic thoughts|evidence for and against)\b', response, re.I
    ):
        issues.append('exercise_without_consent')
    return OutputCheck(valid=not issues, issues=issues, fallback_used=bool(issues))
