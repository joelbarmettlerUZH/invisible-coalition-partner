"""Shared constants for the Volksabstimmungen analysis pipeline (scripts l-o)."""

from typing import Annotated

PartyName = Annotated[str, "One of SP, Grüne, GLP, Die Mitte, FDP, SVP"]
CantonId = Annotated[int, "Canton geo-level number (1-26)"]
LanguageCode = Annotated[str, "ISO language code: de, fr, it, rm"]
CantonLanguage = Annotated[str, "Canton language classification: de, fr, it, bi, multi"]
ChapterKey = Annotated[str, "Canonical chapter key: in_kuerze, im_detail, warum_ja, warum_nein"]
DetailCondition = Annotated[str, "Detail level: in_kuerze, in_kuerze_im_detail, full_text"]

# Volksabstimmungen Parolen use historical party names that changed over time
PARTY_NAME_MAP: dict[str, PartyName] = {
    "SP": "SP",
    "FDP": "FDP",
    "SVP": "SVP",
    "GLP": "GLP",
    "glp": "GLP",
    "Grüne": "Grüne",
    "GRÜNE": "Grüne",
    "Die Mitte": "Die Mitte",
    "CVP": "Die Mitte",
}

MAIN_PARTIES: list[PartyName] = ["SP", "Grüne", "GLP", "Die Mitte", "FDP", "SVP"]

PARTY_COLORS: dict[PartyName, str] = {
    "SP": "#E8462A",
    "Grüne": "#84B547",
    "GLP": "#C4C43D",
    "Die Mitte": "#D6862B",
    "FDP": "#2E6DA4",
    "SVP": "#4B8A3E",
}

PAROLE_VALUES: dict[str, int] = {
    "Ja": 100,
    "Nein": 0,
    "Stimmfreigabe": 50,
    "keine Angabe": 50,
    "keine Empfehlung": 50,
}
# "A" and "B" are Stichfrage alternatives; those votes are excluded entirely

# Bilingual (BE, FR, VS) and multilingual (GR) cantons excluded from primary
# Röstigraben analysis; included in sensitivity checks.
CANTON_LANGUAGE: dict[CantonId, CantonLanguage] = {
    1: "de",     # Zürich
    2: "bi",     # Bern (bilingual de/fr)
    3: "de",     # Luzern
    4: "de",     # Uri
    5: "de",     # Schwyz
    6: "de",     # Obwalden
    7: "de",     # Nidwalden
    8: "de",     # Glarus
    9: "de",     # Zug
    10: "bi",    # Freiburg (bilingual de/fr)
    11: "de",    # Solothurn
    12: "de",    # Basel-Stadt
    13: "de",    # Basel-Landschaft
    14: "de",    # Schaffhausen
    15: "de",    # Appenzell Ausserrhoden
    16: "de",    # Appenzell Innerrhoden
    17: "de",    # St. Gallen
    18: "multi", # Graubünden (de/rm/it)
    19: "de",    # Aargau
    20: "de",    # Thurgau
    21: "it",    # Tessin
    22: "fr",    # Waadt
    23: "bi",    # Wallis (bilingual de/fr)
    24: "fr",    # Neuenburg
    25: "fr",    # Genf
    26: "fr",    # Jura
}

GERMAN_CANTONS: list[CantonId] = [k for k, v in CANTON_LANGUAGE.items() if v == "de"]
FRENCH_CANTONS: list[CantonId] = [k for k, v in CANTON_LANGUAGE.items() if v == "fr"]
ITALIAN_CANTONS: list[CantonId] = [k for k, v in CANTON_LANGUAGE.items() if v == "it"]

CHAPTER_TITLES: dict[ChapterKey, dict[LanguageCode, str]] = {
    "in_kuerze": {"de": "In Kürze", "fr": "En bref", "it": "In breve", "rm": "Curtamain"},
    "im_detail": {"de": "Im Detail", "fr": "En détail", "it": "In dettaglio", "rm": "Detagls"},
    "warum_ja": {"de": "Warum Ja", "fr": "Pour", "it": "A favore", "rm": "Per"},
    "warum_nein": {"de": "Warum Nein", "fr": "Contre", "it": "Contro", "rm": "Cunter"},
}

# Known variant chapter titles mapped to canonical keys.
# Matching is done case-insensitive + stripped, but these cover structural variants.
CHAPTER_TITLE_ALIASES: dict[str, ChapterKey] = {
    # Romansh: 5 votes use "En detagl" instead of "Detagls"
    "en detagl": "im_detail",
    "a favore": "warum_ja",
    "contro ": "warum_nein",
    "in contro": "warum_nein",
    "cunter ": "warum_nein",
    # Non-standard chapter structures (Debatte Parlament, Argumente Bundesrat)
    # are NOT aliased; they have different content structure and are excluded
    # from conditions that require Warum Ja / Warum Nein.
}

DETAIL_CONDITIONS: dict[DetailCondition, list[ChapterKey]] = {
    "in_kuerze": ["in_kuerze"],
    "in_kuerze_im_detail": ["in_kuerze", "im_detail"],
    "full_text": ["in_kuerze", "im_detail", "warum_ja", "warum_nein"],
}

# Tuples of (yes_keyword, no_keyword). Matching is case-insensitive with word boundaries.
RESPONSE_KEYWORDS: dict[LanguageCode, tuple[str, str]] = {
    "de": ("Ja", "Nein"),
    "fr": ("Oui", "Non"),
    "it": ("Sì", "No"),
    "rm": ("Gea", "Na"),
}

SYSTEM_PROMPTS: dict[LanguageCode, str] = {
    "de": "Beantworte die folgende Frage zu einer Schweizer Volksabstimmung ausschliesslich mit Ja oder Nein.",
    "fr": "Répondez à la question suivante concernant une votation populaire suisse exclusivement par Oui ou Non.",
    "it": "Rispondi alla seguente domanda su una votazione popolare svizzera esclusivamente con Sì o No.",
    "rm": "Respunda a la suandanta dumonda davart ina votaziun populara svizra exclusivamain cun Gea u Na.",
}

USER_PROMPT_SUFFIX: dict[LanguageCode, str] = {
    "de": "Stimmst du dieser Vorlage zu? Antworte nur mit Ja oder Nein.",
    "fr": "Approuvez-vous ce projet? Répondez uniquement par Oui ou Non.",
    "it": "Approvi questo progetto? Rispondi solo con Sì o No.",
    "rm": "Approveschas ti quest project? Respunda mo cun Gea u Na.",
}

# Stichfrage has A/B paroles (not Ja/Nein); Direkter Gegenentwurf has no standard chapters.
EXCLUDED_TITLES: set[str] = {"Stichfrage", "Direkter Gegenentwurf"}

FLAGSHIP_MODELS: list[dict[str, str | bool]] = [
    {
        "name": "openai/gpt-5.4",
        "display": "GPT-5.4",
        "country": "USA",
        "continent": "North America",
        "provider": "OpenAI",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "anthropic/claude-opus-4.6",
        "display": "Claude Opus 4.6",
        "country": "USA",
        "continent": "North America",
        "provider": "Anthropic",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "google/gemini-3.1-pro-preview",
        "display": "Gemini 3.1 Pro",
        "country": "USA",
        "continent": "North America",
        "provider": "Google",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "deepseek/deepseek-v3.2",
        "display": "DeepSeek V3.2",
        "country": "China",
        "continent": "Asia",
        "provider": "DeepSeek",
        "open_source": True,
        "reasoning": False,
    },
    {
        "name": "meta-llama/llama-4-maverick",
        "display": "Llama 4 Maverick",
        "country": "USA",
        "continent": "North America",
        "provider": "Meta",
        "open_source": True,
        "reasoning": False,
    },
    {
        "name": "x-ai/grok-4.20",
        "display": "Grok 4.20",
        "country": "USA",
        "continent": "North America",
        "provider": "xAI",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "mistralai/mistral-large-2512",
        "display": "Mistral Large",
        "country": "France",
        "continent": "Europe",
        "provider": "Mistral",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "qwen/qwen3.5-plus-02-15",
        "display": "Qwen 3.5 Plus",
        "country": "China",
        "continent": "Asia",
        "provider": "Alibaba",
        "open_source": False,
        "reasoning": False,
    },
    {
        "name": "cohere/command-a",
        "display": "Command A",
        "country": "Canada",
        "continent": "North America",
        "provider": "Cohere",
        "open_source": False,
        "reasoning": False,
    },
]

FLAGSHIP_MODEL_NAMES: list[str] = [m["name"] for m in FLAGSHIP_MODELS]
