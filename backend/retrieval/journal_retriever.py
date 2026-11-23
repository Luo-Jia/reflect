"""Bounded local lexical retrieval: no model, database, or network required."""
import re
import calendar

STOPWORDS = set('i me my myself we our you your the a an and or to of in on for it is was were am be been have had that this with about from what how can could would should did do does'.split())


def asks_about_journal(message):
    return bool(re.search(r"\b(?:journals?|diar(?:y|ies)|entries|entry|journaled|journaling)\b", message, re.IGNORECASE))


MONTHS = {name.lower(): number for number in range(1, 13)
          for name in (calendar.month_name[number], calendar.month_abbr[number])}
MONTHS['sept'] = 9
MONTH_PATTERN = '(?:' + '|'.join(sorted(MONTHS, key=len, reverse=True)) + ')'
DATE_PATTERNS = [
    re.compile(r'\b(?P<month>' + MONTH_PATTERN + r')\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(?P<year>\d{4})\b)?', re.I),
    re.compile(r'\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>' + MONTH_PATTERN + r')\b\.?(?:,?\s+(?P<year>\d{4})\b)?', re.I),
]


def requested_dates(message):
    exact = set(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', message))
    month_days = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(message):
            month = MONTHS[match['month'].lower()]
            day = int(match['day'])
            if match['year']:
                exact.add(f"{int(match['year']):04d}-{month:02d}-{day:02d}")
            else:
                # Preserve actual matching years rather than inventing one.
                month_days.add(f'{month:02d}-{day:02d}')
    return exact, month_days


def retrieve_journal(message, entries, top_k=3):
    dates, month_days = requested_dates(message)
    if dates or month_days:
        matched = [entry for entry in entries if entry['date'] in dates or entry['date'][5:] in month_days]
        matched.sort(key=lambda entry: (entry['date'], entry['entry_id']), reverse=True)
        return [{**entry, 'content': entry['content'][:1200]} for entry in matched[:top_k]]
    words = set(re.findall(r"[a-z]+", message.lower())) - STOPWORDS
    ranked = []
    for entry in entries:
        tokens = set(re.findall(r"[a-z]+", (entry['title'] + ' ' + entry['content']).lower())) - STOPWORDS
        overlap = len(words & tokens)
        if overlap:
            ranked.append((overlap, entry['date'], entry['entry_id'], entry))
    ranked.sort(key=lambda row: row[:3], reverse=True)
    if not ranked and asks_about_journal(message):
        selected = sorted(entries, key=lambda entry: (entry['date'], entry['entry_id']), reverse=True)[:top_k]
    else:
        selected = [row[3] for row in ranked[:top_k]]
    return [{**entry, 'content': entry['content'][:1200]} for entry in selected]
