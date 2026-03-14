"""
bias_lexicon.py
Comprehensive bias word dictionary with categories, severity levels,
and neutral alternative suggestions.

Categories:
  - gender    : masculine/feminine coded language
  - age       : words that favour or exclude certain age groups
  - ability   : language that disadvantages people with disabilities
  - culture   : culturally exclusive language

Severity:
  1 = mild (slightly exclusionary, often unintentional)
  2 = moderate (clearly gendered / exclusionary)
  3 = high (strongly exclusionary, legally risky)
"""

BIAS_LEXICON = {

    # ── GENDER BIAS — Masculine-coded ──────────────────────────────────────────
    "aggressive":       {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "assertive"},
    "dominant":         {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "influential"},
    "competitive":      {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "goal-oriented"},
    "ninja":            {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "expert"},
    "rockstar":         {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "high performer"},
    "guru":             {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "specialist"},
    "warrior":          {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "dedicated professional"},
    "champion":         {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "leader"},
    "conquer":          {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "achieve"},
    "driven":           {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "motivated"},
    "fearless":         {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "confident"},
    "strong":           {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "capable"},
    "manpower":         {"category": "gender", "severity": 3, "type": "masculine",
                         "neutral": "workforce"},
    "manhole":          {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "utility access point"},
    "chairman":         {"category": "gender", "severity": 3, "type": "masculine",
                         "neutral": "chairperson"},
    "salesman":         {"category": "gender", "severity": 3, "type": "masculine",
                         "neutral": "sales representative"},
    "businessman":      {"category": "gender", "severity": 3, "type": "masculine",
                         "neutral": "business professional"},
    "he":               {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "they"},
    "his":              {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "their"},
    "him":              {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "them"},
    "brotherhood":      {"category": "gender", "severity": 2, "type": "masculine",
                         "neutral": "community"},
    "headcount":        {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "team size"},
    "outspoken":        {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "communicative"},
    "analytical":       {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "data-driven"},
    "independent":      {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "self-directed"},
    "self-reliant":     {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "self-sufficient"},
    "decisive":         {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "clear-thinking"},
    "determined":       {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "persistent"},
    "boast":            {"category": "gender", "severity": 1, "type": "masculine",
                         "neutral": "demonstrate"},

    # ── GENDER BIAS — Feminine-coded ─────────────────────────────────────────
    "nurturing":        {"category": "gender", "severity": 2, "type": "feminine",
                         "neutral": "supportive"},
    "warm":             {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "personable"},
    "caring":           {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "attentive"},
    "sensitive":        {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "empathetic"},
    "gentle":           {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "considerate"},
    "cheerful":         {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "positive"},
    "cooperative":      {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "collaborative"},
    "interpersonal":    {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "communication"},
    "hostess":          {"category": "gender", "severity": 3, "type": "feminine",
                         "neutral": "event coordinator"},
    "stewardess":       {"category": "gender", "severity": 3, "type": "feminine",
                         "neutral": "flight attendant"},
    "receptionist":     {"category": "gender", "severity": 1, "type": "feminine",
                         "neutral": "front desk coordinator"},

    # ── AGE BIAS ──────────────────────────────────────────────────────────────
    "digital native":   {"category": "age", "severity": 3, "type": "youth-bias",
                         "neutral": "comfortable with digital tools"},
    "young":            {"category": "age", "severity": 3, "type": "youth-bias",
                         "neutral": "energetic"},
    "energetic":        {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "motivated"},
    "fresh graduate":   {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "entry-level candidate"},
    "recent graduate":  {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "entry-level candidate"},
    "junior":           {"category": "age", "severity": 1, "type": "youth-bias",
                         "neutral": "entry-level"},
    "youthful":         {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "enthusiastic"},
    "dynamic":          {"category": "age", "severity": 1, "type": "youth-bias",
                         "neutral": "adaptable"},
    "up-and-coming":    {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "emerging professional"},
    "hip":              {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "contemporary"},
    "overqualified":    {"category": "age", "severity": 3, "type": "age-bias",
                         "neutral": "(remove — assess fit objectively)"},
    "tech-savvy":       {"category": "age", "severity": 1, "type": "youth-bias",
                         "neutral": "technically proficient"},
    "fast-paced":       {"category": "age", "severity": 1, "type": "youth-bias",
                         "neutral": "high-output environment"},
    "agile mindset":    {"category": "age", "severity": 1, "type": "youth-bias",
                         "neutral": "adaptable"},
    "2-3 years":        {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "demonstrate relevant experience"},
    "0-2 years":        {"category": "age", "severity": 2, "type": "youth-bias",
                         "neutral": "demonstrate relevant experience"},

    # ── ABILITY BIAS ──────────────────────────────────────────────────────────
    "physically fit":   {"category": "ability", "severity": 3, "type": "physical",
                         "neutral": "able to meet the physical demands of the role"},
    "able-bodied":      {"category": "ability", "severity": 3, "type": "physical",
                         "neutral": "meets physical requirements of the role"},
    "must be able to stand": {"category": "ability", "severity": 2, "type": "physical",
                         "neutral": "role may require extended periods of standing"},
    "native speaker":   {"category": "ability", "severity": 3, "type": "language",
                         "neutral": "fluent in English"},
    "perfect english":  {"category": "ability", "severity": 3, "type": "language",
                         "neutral": "strong written and verbal communication skills"},
    "flawless english": {"category": "ability", "severity": 3, "type": "language",
                         "neutral": "strong written and verbal communication skills"},
    "articulate":       {"category": "ability", "severity": 2, "type": "language",
                         "neutral": "clear communicator"},
    "well-spoken":      {"category": "ability", "severity": 2, "type": "language",
                         "neutral": "strong verbal communication"},
    "no gaps":          {"category": "ability", "severity": 3, "type": "disability",
                         "neutral": "(remove — gaps may reflect caregiving or health)"},

    # ── CULTURAL / EXCLUSIONARY BIAS ─────────────────────────────────────────
    "culture fit":      {"category": "culture", "severity": 3, "type": "culture",
                         "neutral": "values alignment"},
    "culture add":      {"category": "culture", "severity": 1, "type": "culture",
                         "neutral": "values alignment"},
    "ivy league":       {"category": "culture", "severity": 3, "type": "prestige",
                         "neutral": "accredited university"},
    "prestigious university": {"category": "culture", "severity": 2, "type": "prestige",
                         "neutral": "accredited university"},
    "top-tier school":  {"category": "culture", "severity": 2, "type": "prestige",
                         "neutral": "accredited university"},
    "english-medium":   {"category": "culture", "severity": 2, "type": "language",
                         "neutral": "English-language education"},
    "local candidate":  {"category": "culture", "severity": 2, "type": "origin",
                         "neutral": "(remove — assess availability objectively)"},
    "indian origin":    {"category": "culture", "severity": 3, "type": "origin",
                         "neutral": "(remove — illegal to specify)"},
    "christian":        {"category": "culture", "severity": 3, "type": "religion",
                         "neutral": "(remove — illegal to specify)"},
    "brahmin":          {"category": "culture", "severity": 3, "type": "caste",
                         "neutral": "(remove — illegal to specify)"},
    "family man":       {"category": "gender", "severity": 3, "type": "masculine",
                         "neutral": "(remove — irrelevant to job performance)"},
    "housewife":        {"category": "gender", "severity": 3, "type": "feminine",
                         "neutral": "(remove — irrelevant to job performance)"},
}

# ── Category metadata ─────────────────────────────────────────────────────────
CATEGORY_INFO = {
    "gender": {
        "label"      : "Gender Bias",
        "description": "Language that implies a preferred gender for the role",
        "color"      : "#DD8452",
        "icon"       : "⚥",
    },
    "age": {
        "label"      : "Age Bias",
        "description": "Language that favours or excludes certain age groups",
        "color"      : "#4C72B0",
        "icon"       : "📅",
    },
    "ability": {
        "label"      : "Ability Bias",
        "description": "Language that disadvantages people with disabilities",
        "color"      : "#55A868",
        "icon"       : "♿",
    },
    "culture": {
        "label"      : "Cultural Bias",
        "description": "Language that creates cultural or socioeconomic barriers",
        "color"      : "#C44E52",
        "icon"       : "🌍",
    },
}

SEVERITY_LABELS = {
    1: ("Low",      "#55A868"),
    2: ("Moderate", "#DD8452"),
    3: ("High",     "#C44E52"),
}


def get_all_bias_words():
    """Return flat list of all biased words."""
    return list(BIAS_LEXICON.keys())


def get_neutral(word):
    """Return neutral alternative for a biased word."""
    entry = BIAS_LEXICON.get(word.lower())
    return entry["neutral"] if entry else None


def get_category(word):
    """Return category of a biased word."""
    entry = BIAS_LEXICON.get(word.lower())
    return entry["category"] if entry else None


def get_severity(word):
    """Return severity (1-3) of a biased word."""
    entry = BIAS_LEXICON.get(word.lower())
    return entry["severity"] if entry else 0
