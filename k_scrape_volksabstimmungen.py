import json
import time
import requests


BASE_URL = "https://app.statistik.zh.ch/wahlen_abstimmungen/data_prod/"
ARCHIVE_URL = BASE_URL + "archive/0_0_archive.json"


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL, return None on failure."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    print(f"  WARNING: {response.status_code} for {url}")
    return None


def detect_chapter_language(title: str) -> str:
    """Detect the language of a chapter based on its title."""
    title_lower = title.strip().lower()
    fr_titles = {"en bref", "en détail", "pour", "contre", "contre-projet",
                 "débat parlement", "question subsidiaire", "arguments conseil fédéral"}
    it_titles = {"in breve", "in dettaglio", "a favore", "contro", "controprogetto",
                 "in contro", "dibattiti parlamento", "dibattito parlamento",
                 "domanda risolutiva", "gli argomenti consiglio federale",
                 "le deliberazioni parlamento"}
    rm_titles = {"curtamain", "en detagl", "detagls", "per", "cunter", "cuntraproposta",
                 "debatta parlament", "dumonda decisiva", "arguments cussegl federal"}
    if title_lower in fr_titles:
        return "fr"
    if title_lower in it_titles:
        return "it"
    if title_lower in rm_titles:
        return "rm"
    return "de"


def extract_vorlagen_text(vorlagen_raw: list[dict]) -> dict:
    """Extract multilingual text content from Vorlagen.json entries.

    The API returns all languages' chapters under a single langKey="de" entry,
    so we detect the actual language from each chapter's title.

    Returns a dict keyed by vorlagenId, each containing title and text per language.
    """
    by_id = {}
    for entry in vorlagen_raw:
        vid = entry["vorlagenId"]

        if vid not in by_id:
            by_id[vid] = {"vorlagenId": vid, "titles": {}, "texts": {}}

        by_id[vid]["titles"]["de"] = entry.get("vorlagenTitel", "")

        # Extract readable text from erlaeuterungen chapters, split by language
        for erl in entry.get("erlaeuterungen", []):
            for kapitel in erl.get("kapitel", []):
                chapter_title = kapitel.get("text", "")
                lang = detect_chapter_language(chapter_title)
                chapter = {"title": chapter_title, "content": []}
                for komp in kapitel.get("komponenten", []):
                    typ = komp.get("typ")
                    if typ == "text" and komp.get("text"):
                        text_content = komp["text"].get("text", "")
                        if text_content:
                            chapter["content"].append(text_content)
                    elif typ == "title" and komp.get("title"):
                        title_text = komp["title"].get("text", "")
                        if title_text:
                            chapter["content"].append(title_text)
                if chapter["content"]:
                    if lang not in by_id[vid]["texts"]:
                        by_id[vid]["texts"][lang] = []
                    by_id[vid]["texts"][lang].append(chapter)

    return by_id


def extract_canton_results(resultate_raw: dict, vorlagen_id: int) -> list[dict]:
    """Extract per-canton results for a given Vorlage from Resultate.json."""
    for vorlage in resultate_raw.get("schweiz", {}).get("vorlagen", []):
        if vorlage.get("vorlagenId") == vorlagen_id:
            cantons = []
            for kanton in vorlage.get("kantone", []):
                resultat = kanton.get("resultat") or {}
                cantons.append({
                    "geoLevelnummer": kanton.get("geoLevelnummer"),
                    "name": kanton.get("geoLevelname", ""),
                    "jaStimmenInProzent": resultat.get("jaStimmenInProzent"),
                    "jaStimmenAbsolut": resultat.get("jaStimmenAbsolut"),
                    "neinStimmenAbsolut": resultat.get("neinStimmenAbsolut"),
                    "stimmbeteiligungInProzent": resultat.get("stimmbeteiligungInProzent"),
                    "anzahlStimmberechtigte": resultat.get("anzahlStimmberechtigte"),
                })
            return cantons
    return []


def extract_parolen(parolen_raw: dict, vorlagen_id: int) -> list[dict]:
    """Extract national (ch) party parolen for a given Vorlage."""
    for vorlage in parolen_raw.get("vorlagen", []):
        if vorlage["vorlagenId"] == vorlagen_id:
            parolen = vorlage.get("parolen", {})
            ch_parolen = parolen.get("ch") or []
            rat_parolen = parolen.get("rat") or []
            return {
                "parties": [
                    {
                        "name": p["name"],
                        "parole": p["parole"],
                        "color": p["color"],
                    }
                    for p in ch_parolen
                    if p.get("parole")
                ],
                "councils": [
                    {
                        "name": p["name"],
                        "parole": p["parole"],
                    }
                    for p in rat_parolen
                    if p.get("parole")
                ],
            }
    return {"parties": [], "councils": []}


def main():
    print("Fetching archive index...")
    archive = fetch_json(ARCHIVE_URL)
    if not archive:
        raise RuntimeError("Failed to fetch archive")

    geschaefte = archive["geschaefte"]
    print(f"Found {len(geschaefte)} Geschäfte in archive")

    # Group by abstimmtag to avoid fetching the same Vorlagen/Parolen URL multiple times
    # (multiple Geschäfte on the same day share the same URL directory)
    by_date = {}
    for g in geschaefte:
        tag = g["abstimmtag"]
        if tag not in by_date:
            by_date[tag] = []
        by_date[tag].append(g)

    print(f"Spanning {len(by_date)} voting dates from {min(by_date)} to {max(by_date)}")

    all_vorlagen = []

    for i, (tag, geschaefte_list) in enumerate(sorted(by_date.items())):
        # Derive the URL directory from the first geschaeft's urlVorlagen
        url_vorlagen_path = geschaefte_list[0].get("urlVorlagen")
        if not url_vorlagen_path:
            print(f"  [{i+1}/{len(by_date)}] {tag}: no urlVorlagen, skipping")
            continue

        # Construct base directory: geschaefte/0_0_YYYYMMDD/
        base_dir = url_vorlagen_path.rsplit("/", 1)[0]
        vorlagen_url = BASE_URL + url_vorlagen_path
        parolen_url = BASE_URL + base_dir + "/Parolen.json"
        resultate_url = BASE_URL + base_dir + "/Resultate.json"

        print(f"  [{i+1}/{len(by_date)}] {tag}: fetching Vorlagen + Parolen + Resultate...")

        vorlagen_raw = fetch_json(vorlagen_url)
        parolen_raw = fetch_json(parolen_url)
        resultate_raw = fetch_json(resultate_url)

        if not vorlagen_raw:
            print(f"    No Vorlagen data for {tag}")
            continue

        # vorlagen_raw is a dict with a "vorlagen" key containing the list
        vorlagen_list = vorlagen_raw.get("vorlagen", vorlagen_raw)
        if isinstance(vorlagen_list, dict):
            vorlagen_list = [vorlagen_list]

        vorlagen_by_id = extract_vorlagen_text(vorlagen_list)

        for g in geschaefte_list:
            vid = g["vorlagenId"]
            vorlage_info = vorlagen_by_id.get(vid, {})
            resultat = g.get("resultat") or {}

            parolen = {"parties": [], "councils": []}
            if parolen_raw:
                parolen = extract_parolen(parolen_raw, vid)

            kantone = []
            if resultate_raw:
                kantone = extract_canton_results(resultate_raw, vid)

            entry = {
                "vorlagenId": vid,
                "abstimmtag": tag,
                "titel": g.get("titel", ""),
                "titles": vorlage_info.get("titles", {}),
                "geschaeftsTypId": g.get("geschaeftsTypId"),
                "geschaeftsSubTypId": g.get("geschaeftsSubTypId"),
                "geschaeftsArtId": g.get("geschaeftsArtId"),
                "resultat": {
                    "jaStimmenInProzent": resultat.get("jaStimmenInProzent"),
                    "jaStimmenAbsolut": resultat.get("jaStimmenAbsolut"),
                    "neinStimmenAbsolut": resultat.get("neinStimmenAbsolut"),
                    "stimmbeteiligungInProzent": resultat.get("stimmbeteiligungInProzent"),
                    "anzahlStimmberechtigte": resultat.get("anzahlStimmberechtigte"),
                } if resultat else None,
                "kantone": kantone,
                "parolen": parolen,
                "texts": vorlage_info.get("texts", {}),
            }
            all_vorlagen.append(entry)

        # Be polite to the server
        time.sleep(0.3)

    # Keep only Vorlagen that have both rich text and party parolen
    all_vorlagen = [
        v for v in all_vorlagen
        if v["texts"] and v["parolen"]["parties"]
    ]

    # Save
    output_path = "./data/volksabstimmungen/volksabstimmungen.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_vorlagen, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_vorlagen)} Vorlagen to {output_path}")

    # Print summary stats
    dates = sorted(set(v["abstimmtag"] for v in all_vorlagen))
    print(f"Date range: {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
