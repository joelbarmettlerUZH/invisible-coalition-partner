#!/usr/bin/env python3
"""
Query flagship LLMs on Swiss Volksabstimmungen (popular votes).

For each of 9 flagship models × 48 usable votes × 4 languages × 3 detail
conditions, sends the referendum summary text and collects a binary Ja/Nein
answer. Results are stored incrementally in a JSON file.

Usage:
    uv run python l_generate_volksabstimmungen_dataset.py
    uv run python l_generate_volksabstimmungen_dataset.py --dry-run   # print prompts, don't call API
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

from _volksabstimmungen_constants import (
    CHAPTER_TITLE_ALIASES,
    CHAPTER_TITLES,
    DETAIL_CONDITIONS,
    EXCLUDED_TITLES,
    FLAGSHIP_MODELS,
    RESPONSE_KEYWORDS,
    SYSTEM_PROMPTS,
    USER_PROMPT_SUFFIX,
)

load_dotenv()

VOLKSABSTIMMUNGEN_FILE = "./data/volksabstimmungen/volksabstimmungen.json"
OUTPUT_FILE = "./data/answers/volksabstimmungen_model_answers.json"

BASE_PARAMS = {
    "temperature": 0.0,
    "seed": 42,
}

LANGUAGES = ["de", "fr", "it", "rm"]


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_html(html_text):
    """Strip HTML tags from text content."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ").strip()


def get_usable_votes(data):
    """Filter out Stichfrage and votes without party Parolen."""
    return [v for v in data if v["titel"] not in EXCLUDED_TITLES and v["parolen"]["parties"]]


def _build_chapter_lookup(texts, lang):
    """
    Build a mapping from canonical chapter key to content list.

    Matches by exact title, then case-insensitive+stripped, then known aliases.
    """
    # Build reverse lookup: for each chapter_key, what title do we expect?
    expected = {}  # canonical_title -> chapter_key
    for chapter_key, titles_by_lang in CHAPTER_TITLES.items():
        if lang in titles_by_lang:
            expected[titles_by_lang[lang]] = chapter_key

    lookup = {}  # chapter_key -> content list

    for chapter in texts:
        raw_title = chapter["title"]
        content = chapter["content"]

        # 1. Exact match
        if raw_title in expected:
            lookup[expected[raw_title]] = content
            continue

        # 2. Case-insensitive + stripped match
        normalized = raw_title.lower().strip()
        matched = False
        for exp_title, chapter_key in expected.items():
            if exp_title.lower().strip() == normalized:
                lookup[chapter_key] = content
                matched = True
                break

        if matched:
            continue

        # 3. Known alias match
        alias_key = CHAPTER_TITLE_ALIASES.get(normalized)
        if alias_key is not None:
            lookup[alias_key] = content

    return lookup


def extract_text_for_condition(vote, lang, condition):
    """
    Extract and clean the text content for a given vote, language, and detail condition.

    Returns the cleaned text string, or None if the required chapters are missing.
    """
    chapters_needed = DETAIL_CONDITIONS[condition]
    texts = vote.get("texts", {}).get(lang, [])
    chapter_lookup = _build_chapter_lookup(texts, lang)

    parts = []
    for chapter_key in chapters_needed:
        content = chapter_lookup.get(chapter_key)
        if content is None:
            return None  # Required chapter missing

        # Clean and join paragraphs
        cleaned_paragraphs = [clean_html(p) for p in content if p.strip()]
        parts.append("\n".join(cleaned_paragraphs))

    return "\n\n".join(parts)


def build_prompt(vote, lang, condition):
    """Build the system and user messages for a Volksabstimmungen query."""
    text = extract_text_for_condition(vote, lang, condition)
    if text is None:
        return None, None

    system_msg = SYSTEM_PROMPTS[lang]
    user_msg = (
        f"Volksabstimmung vom {vote['abstimmtag']}: {vote['titel']}\n\n"
        f"{text}\n\n"
        f"{USER_PROMPT_SUFFIX[lang]}"
    )

    return system_msg, user_msg


def parse_response(response_text, lang):
    """
    Parse a model response to extract Ja/Nein (or language equivalent).
    Returns 100 (Ja), 0 (Nein), or -1 (refused/unparseable).

    Strategy: Models are prompted to respond with a single word. We only parse
    responses that are clearly a single-word answer or that start with the answer
    keyword. Longer responses that contain both yes and no keywords (e.g.,
    "I would vote yes if X and no if Y") are treated as refused (-1) since we
    cannot reliably determine the model's position.
    """
    if not response_text:
        return -1

    text = response_text.strip()
    text_lower = text.lower()
    yes_kw, no_kw = RESPONSE_KEYWORDS[lang]

    # Define strict patterns per language
    if lang == "it":
        # Italian: only match "Sì" (with accent) as yes. Plain "si" is too ambiguous
        # (reflexive pronoun, impersonal "one"). "No" is unambiguous.
        yes_pattern = r"\bsì\b"
        no_pattern = r"\bno\b"
    elif lang == "rm":
        # Romansh: "Gea" and "Na" are short — require start of response
        yes_pattern = r"\bgea\b"
        no_pattern = r"\bna\b"
    else:
        yes_pattern = r"\b" + re.escape(yes_kw.lower()) + r"\b"
        no_pattern = r"\b" + re.escape(no_kw.lower()) + r"\b"

    # Phase 1: Check if the response is very short (≤ 3 words). These are almost
    # certainly direct answers like "Ja", "Nein, danke", "Oui, bien sûr".
    words = text.split()
    if len(words) <= 3:
        yes_match = re.search(yes_pattern, text_lower)
        no_match = re.search(no_pattern, text_lower)

        if yes_match and no_match:
            return -1  # Ambiguous even in a very short response
        if yes_match:
            return 100
        if no_match:
            return 0

    # Phase 2: For any longer response, ONLY match if it starts with the answer
    # keyword (possibly followed by punctuation/space). This avoids false positives
    # from hedging ("I would vote yes if... but no if...") or Italian "si" as pronoun.
    start_yes = re.match(r"^\s*" + yes_pattern + r"[\s.,!;:\-]", text_lower)
    start_no = re.match(r"^\s*" + no_pattern + r"[\s.,!;:\-]", text_lower)

    # Also check exact single-word match (response is just "Ja." or "Nein")
    stripped = text_lower.strip().rstrip(".")
    if stripped == yes_kw.lower() or (lang == "it" and stripped == "sì"):
        return 100
    if stripped == no_kw.lower():
        return 0

    if start_yes and not start_no:
        return 100
    if start_no and not start_yes:
        return 0

    # Phase 3: Check if the response ENDS with the answer keyword.
    # Chain-of-thought models reason at length then give a final answer.
    tail = text_lower[-80:]
    tail_yes = re.findall(yes_pattern, tail)
    tail_no = re.findall(no_pattern, tail)
    tail_all = tail_yes + tail_no

    if tail_all:
        tail_unique_yes = len(set(tail_yes))
        tail_unique_no = len(set(tail_no))
        # Only trust if the tail has exactly one type of keyword (the conclusion)
        if tail_unique_yes > 0 and tail_unique_no == 0:
            return 100
        if tail_unique_no > 0 and tail_unique_yes == 0:
            return 0
        # Both present in tail but check for explicit "Antwort"/"boxed" markers
        if "antwort" in tail or "boxed" in tail or "réponse" in tail or "risposta" in tail:
            # Last keyword wins
            last_kw_yes = tail.rfind(tail_yes[-1]) if tail_yes else -1
            last_kw_no = tail.rfind(tail_no[-1]) if tail_no else -1
            if last_kw_yes > last_kw_no:
                return 100
            if last_kw_no > last_kw_yes:
                return 0

    return -1  # Unparseable / refused / ambiguous


class VolksabstimmungenProcessor:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        if not dry_run:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        self.output_file = OUTPUT_FILE
        self.results = self._load_existing_results()

    def _load_existing_results(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_results(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def _get_or_create_model_result(self, model_config):
        """Find or create the result entry for a model."""
        for result in self.results:
            if result["name"] == model_config["name"]:
                return result

        new_result = {
            "name": model_config["name"],
            "display": model_config["display"],
            "country": model_config["country"],
            "continent": model_config["continent"],
            "provider": model_config["provider"],
            "open_source": model_config["open_source"],
            "reasoning": model_config["reasoning"],
            **BASE_PARAMS,
            "timestamp": datetime.now().isoformat(),
            "conditions": {},
        }
        self.results.append(new_result)
        return new_result

    def _is_already_answered(self, model_result, condition, lang, vorlage_id):
        """Check if a specific (condition, language, vote) has been answered."""
        cond_data = model_result.get("conditions", {}).get(condition, {})
        lang_data = cond_data.get(lang, {})
        for answer in lang_data.get("answers", []):
            if answer["vorlagenId"] == vorlage_id:
                return True
        return False

    def _query_model(self, model_name, system_msg, user_msg):
        """Send a query to the model via OpenRouter and return the response."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=BASE_PARAMS["temperature"],
            seed=BASE_PARAMS["seed"],
        )

        response_text = completion.choices[0].message.content
        usage = completion.usage

        return {
            "raw_value": response_text,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "metadata": {
                "system_fingerprint": completion.system_fingerprint,
                "finish_reason": completion.choices[0].finish_reason,
                "response_created": completion.created,
            },
        }

    def process_all(self, votes):
        """Process all models × votes × languages × conditions."""
        total_calls = len(FLAGSHIP_MODELS) * len(votes) * len(LANGUAGES) * len(DETAIL_CONDITIONS)
        completed = 0
        skipped = 0

        for model_config in FLAGSHIP_MODELS:
            model_name = model_config["name"]
            model_result = self._get_or_create_model_result(model_config)
            print(f"\n{'='*60}")
            print(f"Model: {model_config['display']} ({model_name})")

            for condition in DETAIL_CONDITIONS:
                # Ensure condition structure exists
                if condition not in model_result.get("conditions", {}):
                    model_result.setdefault("conditions", {})[condition] = {}

                for lang in LANGUAGES:
                    # Ensure language structure exists
                    if lang not in model_result["conditions"][condition]:
                        model_result["conditions"][condition][lang] = {"answers": []}

                    for vote in votes:
                        vorlage_id = vote["vorlagenId"]

                        if self._is_already_answered(model_result, condition, lang, vorlage_id):
                            skipped += 1
                            continue

                        system_msg, user_msg = build_prompt(vote, lang, condition)
                        if system_msg is None:
                            print(f"  SKIP {lang}/{condition}: missing text for {vote['titel'][:40]}")
                            skipped += 1
                            continue

                        if self.dry_run:
                            print(f"  [DRY RUN] {lang}/{condition}: {vote['titel'][:50]}")
                            print(f"    System: {system_msg[:80]}...")
                            print(f"    User: {user_msg[:120]}...")
                            completed += 1
                            if completed >= 5:
                                print(f"\n  ... (showing first 5 of {total_calls} calls)")
                                return
                            continue

                        try:
                            result = self._query_model(model_name, system_msg, user_msg)
                            value = parse_response(result["raw_value"], lang)

                            answer = {
                                "vorlagenId": vorlage_id,
                                "raw_value": result["raw_value"],
                                "value": value,
                                "usage": result["usage"],
                                "metadata": result["metadata"],
                            }

                            model_result["conditions"][condition][lang]["answers"].append(answer)
                            completed += 1

                            value_str = {100: "Ja", 0: "Nein", -1: "REFUSED"}.get(value, str(value))
                            print(
                                f"  [{completed}/{total_calls - skipped}] "
                                f"{lang}/{condition}: {vote['titel'][:40]} → {value_str}"
                            )

                            # Save after each answer for resume capability
                            self._save_results()

                            # Brief pause to respect rate limits
                            time.sleep(0.3)

                        except Exception as e:
                            print(f"  ERROR {lang}/{condition}: {vote['titel'][:40]} → {e}")
                            # Save progress even on error
                            self._save_results()
                            continue

        print(f"\nDone. Completed: {completed}, Skipped (already done): {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Query LLMs on Swiss Volksabstimmungen")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    data = load_json(VOLKSABSTIMMUNGEN_FILE)
    votes = get_usable_votes(data)
    print(f"Loaded {len(votes)} usable votes")
    print(f"Models: {len(FLAGSHIP_MODELS)}")
    print(f"Languages: {LANGUAGES}")
    print(f"Conditions: {list(DETAIL_CONDITIONS.keys())}")
    print(f"Total API calls: {len(FLAGSHIP_MODELS) * len(votes) * len(LANGUAGES) * len(DETAIL_CONDITIONS)}")

    processor = VolksabstimmungenProcessor(dry_run=args.dry_run)
    processor.process_all(votes)


if __name__ == "__main__":
    main()
