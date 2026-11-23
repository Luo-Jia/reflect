"""Explicit, in-memory conversation state; never persisted automatically."""
from dataclasses import dataclass, field, asdict
import re


@dataclass
class SessionState:
    profile_id: str = "local-user"
    topic: str | None = None
    pending_question: str | None = None
    proposed_exercise: str | None = None
    accepted_exercise: str | None = None
    consent_status: str = "unknown"
    turn_count: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    earlier_user_messages: list[str] = field(default_factory=list)
    journal_entries: list[dict] = field(default_factory=list)

    def resolve_reply(self, message, context):
        reply = message.strip().lower().rstrip(".!?")
        # A short acceptance is meaningful only for our explicit pending offer.
        accepted = reply in {"yes", "yes please", "sure", "okay", "ok", "let's try"}
        refused = reply in {"no", "no thanks", "not now", "stop", "stop please"}
        refused = refused or bool(re.search(r"\b(don't|do not) (want to|want an|want any|continue)\b", reply))
        scoped_acceptance = bool(accepted and self.proposed_exercise and self.pending_question)
        self.accepted_exercise = self.proposed_exercise if scoped_acceptance and not refused else None
        self.consent_status = "declined" if refused else (
            "granted" if scoped_acceptance or context.consent_to_exercise else "unknown"
        )
        context.consent_to_exercise = self.consent_status == "granted"
        self.pending_question = None
        self.proposed_exercise = None
        if context.automatic_thought:
            self.topic = context.automatic_thought
        elif self.topic is None:
            self.topic = message[:200]

    def record(self, message, response, offer=None):
        self.turn_count += 1
        questions = re.findall(r"[^.!?\n]+\?", response)
        self.pending_question = questions[-1].strip() if questions else None
        self.proposed_exercise = offer
        self.history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ])
        # Keep bounded, verbatim excerpts rather than inventing inferred facts.
        # This recap stays in memory and is never saved to the profile.
        for item in self.history[:-12]:
            if item["role"] == "user":
                self.earlier_user_messages.append(item["content"][:300])
        self.earlier_user_messages = self.earlier_user_messages[-8:]
        self.history = self.history[-12:]

    def summary(self):
        data = asdict(self)
        data.pop("history")
        data.pop("journal_entries")
        return data
