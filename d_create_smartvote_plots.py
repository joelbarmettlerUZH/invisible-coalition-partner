import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA


### Helper functions ###

def load_json(path):
    """Load JSON data from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_models(path, *, reasoning=None, open_source=None, country=None, continent=None):
    models = load_json(path)
    if reasoning is not None:
        models = [m for m in models if m.get("reasoning") == reasoning]
    if open_source is not None:
        models = [m for m in models if m.get("open_source") == open_source]
    if country is not None:
        models = [m for m in models if m.get("country") == country]
    if continent is not None:
        models = [m for m in models if m.get("continent") == continent]
    return models

def load_questionnaire(path):
    """
    Load the questionnaire JSON and return a sorted list of all question IDs.
    (Assumes questions are spread over categories.)
    """
    data = load_json(path)
    question_ids = set()
    for category in data.get("categories", []):
        for question in category.get("questions", []):
            question_ids.add(question["id"])
    return sorted(list(question_ids))


def build_answer_mapping(answer_list):
    """Convert a list of answer objects into a mapping: questionId -> numeric answer."""
    mapping = {}
    for answer in answer_list:
        try:
            val = float(answer["value"])
            if val == -1:
                continue  # Skip unanswered questions.
            mapping[answer["questionId"]] = val
        except (KeyError, ValueError, TypeError):
            continue
    return mapping


def build_answer_vector(answer_mapping, question_ids, default=50.0):
    """
    Return a numpy vector of answers for the given question_ids.
    Missing answers are imputed with the default (neutral 50).
    """
    vec = []
    for qid in question_ids:
        vec.append(answer_mapping.get(qid, default))
    return np.array(vec)


def compute_agreement_percentage(vec1, vec2):
    """
    Given two answer vectors, compute the average absolute difference,
    and convert it to an agreement percentage where:
      0 difference -> 100% agreement,
      100 difference -> 0% agreement.
    """
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return None
    avg_abs_diff = np.mean(np.abs(vec1 - vec2))
    agreement = 100 * (1 - (avg_abs_diff / 100))
    return agreement


def extract_party_info(politicians):
    """
    Extract party information from the politician data.
    Returns a dict mapping party abbreviation to a dict with keys 'name' and 'color'.
    """
    parties = {}
    for pol in politicians:
        party_obj = pol.get("party", {})
        abbrev = party_obj.get("abbreviation", pol.get("partyAbbreviation"))
        name = party_obj.get("name", None)
        color = party_obj.get("color", pol.get("partyColor"))
        if abbrev:
            if abbrev not in parties:
                parties[abbrev] = {"name": name, "color": color}
    return parties




def create_pca_plot(pol_vectors, model_vectors, pol_party, party_info, party_order, model_marker):
    """
    Creates and shows a PCA plot for all answer vectors (politicians and LLMs).
    Politicians are plotted with their party's color, and LLMs are plotted in black with
    distinct marker styles. The legend (showing party short names and LLM markers) is placed
    outside the main plot area.
    """
    # Gather all vectors and labels.
    participant_labels = []  # List of tuples: (type, id)
    vectors = []
    for pid, vec in pol_vectors.items():
        participant_labels.append(("politician", pid))
        vectors.append(vec)
    for model_name, vec in model_vectors.items():
        participant_labels.append(("model", model_name))
        vectors.append(vec)

    data_matrix = np.vstack(vectors)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data_matrix)

    # Create a wider figure.
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each point.
    for (ptype, label), coord in zip(participant_labels, coords):
        if ptype == "politician":
            party = pol_party[label]
            color = party_info.get(party, {}).get("color", "gray")
            ax.scatter(coord[0] * -1, coord[1], color=color, s=50, alpha=0.7)
        else:  # model
            marker = model_marker[label]
            ax.scatter(coord[0] * -1, coord[1], color="black", marker=marker, s=100, edgecolor="w")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Overall Answer Vectors")

    # Create legend entries.
    # LLM legend:
    llm_handles = [Line2D([0], [0], marker=model_marker[m], color="black", linestyle="",
                          markersize=8, label=m) for m in model_vectors.keys()]
    # Party legend (only short names):
    party_handles = [Patch(facecolor=party_info[p]["color"], label=p) for p in party_order]
    # Combine legends.
    all_handles = llm_handles + party_handles
    ax.legend(handles=all_handles, title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def create_agreement_boxplot(model_name, model_vector, party_order, party_to_pol_ids, pol_vectors, party_info):
    """
    Creates a boxplot for a given LLM showing the distribution of agreement percentages
    with each party. For each party, the agreement percentage is computed for every politician
    (using the provided model_vector and each politician's vector) and a boxplot is drawn.
    """
    # For each party, compute list of agreement percentages.
    boxplot_data = []
    for party in party_order:
        agreements = []
        for pid in party_to_pol_ids.get(party, []):
            p_vec = pol_vectors[pid]
            agr = compute_agreement_percentage(model_vector, p_vec)
            if agr is not None:
                agreements.append(agr)
        boxplot_data.append(agreements)

    # Create a wider figure.
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(boxplot_data, patch_artist=True, labels=party_order)

    # Color each box with the party color.
    for patch, party in zip(bp["boxes"], party_order):
        patch.set_facecolor(party_info.get(party, {}).get("color", "gray"))

    ax.set_ylim(0, 105)
    ax.set_ylabel("Agreement Percentage")
    ax.set_title(f"Agreement Distribution of {model_name} with Each Party")

    plt.tight_layout()
    plt.show()



def create_spider_chart_for_llm(llm_name, llm_mapping, category_map, default=50.0):
    """
    Create a spider (radar) chart for a given LLM.

    Each question category is represented as an axis. For each category:
      - The maximum possible points = (number of questions in that category × 100).
      - The LLM’s points = sum of its answers for that category (using `default` for missing answers).
      - The percentage score for the category is calculated as:

            percentage = (LLM's total score / maximum possible score) * 100

    This percentage is plotted on the axis (so that 100 means perfect scores in that category).

    Parameters:
      - llm_name: str, the name of the language model.
      - llm_mapping: dict mapping question ID to numeric answer (0–100) for the LLM.
      - category_map: dict mapping category id to a dict with keys "name" and "question_ids".
      - default: default answer value if a question is missing (default is 50).
    """
    category_names = []
    percentages = []

    # Process each category in the questionnaire.
    for cat_id, cat_info in category_map.items():
        qids = cat_info.get("question_ids", [])
        if not qids:
            continue
        # For each question, use the LLM's answer or the default.
        answers = [llm_mapping.get(qid, default) for qid in qids]
        total_score = sum(answers)
        max_score = len(qids) * 100.0
        pct = (total_score / max_score) * 100.0  # This equals the average answer.
        category_names.append(cat_info.get("name", f"Cat {cat_id}"))
        percentages.append(pct)

    # Number of categories (axes)
    N = len(category_names)

    # Compute the angle (in radians) for each axis.
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    # To complete the loop, append the first angle and first value again.
    angles += angles[:1]
    percentages += percentages[:1]

    # Create the radar chart.
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, percentages, linewidth=2, linestyle='solid', label=llm_name)
    ax.fill(angles, percentages, alpha=0.25)

    # Set the category labels (one per axis).
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_names, fontsize=10)

    # Set radial limits: 0 (worst) to 100 (best).
    ax.set_rlabel_position(30)
    ax.set_ylim(0, 100)

    ax.set_title(f"Spider Chart for {llm_name}", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()


def create_1d_spectrum_plot(pol_vectors, pol_party, model_vectors, party_info, party_order, model_marker):
    """
    Create a 1D political spectrum (left vs. right) using PCA with n_components=1.

    For each party, compute the average answer vector from its politicians.
    Then, combine these party-average vectors with the overall LLM vectors,
    and apply PCA (n_components=1) to obtain a single coordinate for each.

    The function plots a horizontal scatter plot:
      - Parties are plotted at y=0, using their party color with text labels.
      - LLMs are plotted at y=0, with different markers (and no inline text).
      - A legend is created for the LLMs.
    """
    # Compute party average vectors.
    party_avg_vectors = {}
    for party in party_order:
        # Get all politician IDs belonging to this party.
        p_ids = [pid for pid, p in pol_party.items() if p == party]
        if p_ids:
            # Stack the vectors and compute the mean.
            vecs = np.vstack([pol_vectors[pid] for pid in p_ids])
            avg_vec = np.mean(vecs, axis=0)
            party_avg_vectors[party] = avg_vec

    # Combine party-average vectors and LLM vectors.
    combined_labels = []  # Each entry: (type, label) where type is "party" or "model"
    combined_vectors = []  # Corresponding answer vectors.

    for party, vec in party_avg_vectors.items():
        combined_labels.append(("party", party))
        combined_vectors.append(vec)

    for model_name, vec in model_vectors.items():
        combined_labels.append(("model", model_name))
        combined_vectors.append(vec)

    combined_matrix = np.vstack(combined_vectors)
    pca = PCA(n_components=1)
    # Flatten to a 1D array.
    coords = pca.fit_transform(combined_matrix).flatten()
    # Multiply by -1 to flip the axis if needed.
    coords = coords * 1

    # Create a horizontal scatter plot.
    fig, ax = plt.subplots(figsize=(12, 3))
    # Turn off the axis.
    ax.axis("off")

    # Remove x-axis ticks and labels.
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Plot each point.
    for (ptype, label), x_coord in zip(combined_labels, coords):
        if ptype == "party":
            color = party_info.get(label, {}).get("color", "gray")
            ax.scatter(x_coord, 0, color=color, s=150, marker="o", edgecolor="k", zorder=3)
            ax.text(x_coord, -0.1, label, ha="center", va="top", fontsize=10, color=color)
        else:  # model: use the marker from model_marker
            marker = model_marker[label]
            ax.scatter(x_coord, 0, color="black", marker=marker, s=50, zorder=3, alpha=0.5)
            # No inline label for models.

    # Draw a horizontal baseline that doesn't extend over the "Left"/"Right" labels.
    # Get current x-axis limits and then shrink the line's extent.
    xmin, xmax = ax.get_xlim()
    line_start = xmin + 20
    line_end = xmax - 20
    ax.hlines(y=0, xmin=line_start, xmax=line_end, color="gray", linestyle="--", linewidth=1)

    # Adjust x-axis limits for some padding.
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(-0.5, 0.5)

    # Annotate the ends of the spectrum.
    ax.text(xmin - 10, -0.02, "Left", ha="left", fontsize=12)
    ax.text(xmax + 15, -0.02, "Right", ha="right", fontsize=12)

    # Remove y-axis ticks.
    ax.set_yticks([])

    # Add a legend for the LLM markers.
    from matplotlib.lines import Line2D
    model_handles = []
    for model_name in model_vectors.keys():
        marker = model_marker[model_name]
        handle = Line2D([0], [0], marker=marker, color="black", linestyle="", label=model_name)
        model_handles.append(handle)
    # Place the legend above the plot.
    ax.legend(handles=model_handles, loc="lower center", fontsize=8, bbox_to_anchor=(0.5, -0.5), ncol=len(model_handles) // 3,
              frameon=False)

    # Title and optional x-axis label.
    ax.set_xlabel("Political Spectrum (Left vs. Right)", fontsize=12)
    ax.set_title("1D Political Spectrum: Parties and LLMs", fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def load_category_map(path):
    """
    Load the questionnaire JSON and return a dictionary mapping category IDs
    to a dict with keys "name" and "question_ids".
    """
    data = load_json(path)
    category_map = {}
    for category in data.get("categories", []):
        cat_id = category.get("id")
        cat_name = category.get("name")
        question_ids = [q["id"] for q in category.get("questions", [])]
        category_map[cat_id] = {"name": cat_name, "question_ids": question_ids}
    return category_map


def compute_party_average_mapping(politician_mappings, question_ids, default=50.0):
    """
    Given a list of politician answer mappings (each mapping: questionId -> answer)
    and the global list of question_ids, compute a new mapping for the party where,
    for each question, the average answer is computed.
    """
    avg_mapping = {}
    for qid in question_ids:
        total = 0.0
        count = 0
        for mapping in politician_mappings:
            # Use the answer if available, otherwise the default.
            total += mapping.get(qid, default)
            count += 1
        avg_mapping[qid] = total / count if count > 0 else default
    return avg_mapping


def create_spider_chart_for_party(party_name, party_mapping, category_map, party_color, default=50.0):
    """
    Create a spider (radar) chart for a given party in its party color.

    Each question category is represented as an axis. For each category:
      - The maximum possible points = (number of questions in that category × 100).
      - The party’s points = sum of its average answers for that category.
      - The percentage score is:

            percentage = (Total points / Maximum possible points) * 100

    This percentage (essentially the average answer percentage) is plotted on the axis.

    Parameters:
      - party_name: str, the short name of the party.
      - party_mapping: dict mapping question ID to average answer for that party.
      - category_map: dict mapping category id to a dict with keys "name" and "question_ids".
      - party_color: str, the color associated with the party (e.g. "#4B8A3E").
      - default: default answer value if a question is missing (default is 50).
    """
    category_names = []
    percentages = []

    # Process each category in the questionnaire.
    for cat_id, cat_info in category_map.items():
        qids = cat_info.get("question_ids", [])
        if not qids:
            continue
        # Use the party's average answer (or default) for each question.
        answers = [party_mapping.get(qid, default) for qid in qids]
        total_score = sum(answers)
        max_score = len(qids) * 100.0
        pct = (total_score / max_score) * 100.0  # Essentially the average answer.
        category_names.append(cat_info.get("name", f"Cat {cat_id}"))
        percentages.append(pct)

    # Number of categories (axes).
    N = len(category_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    # Close the polygon.
    angles += angles[:1]
    percentages += percentages[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    # Plot with the party's color.
    ax.plot(angles, percentages, linewidth=2, linestyle='solid', label=party_name, color=party_color)
    ax.fill(angles, percentages, alpha=0.25, color=party_color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_names, fontsize=10)
    ax.set_rlabel_position(30)
    ax.set_ylim(0, 100)
    ax.set_title(f"Spider Chart for Party {party_name}", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()


def create_correlation_matrix_by_category(category_map, party_to_pol_ids, pol_mapping, model_answers, default=50.0):
    """
    For each category, compute a correlation matrix between the average party responses
    (across its politicians) and each LLM’s responses.

    For each category in category_map:
      - Extract the list of question IDs.
      - For each party (from party_to_pol_ids), compute the average answer for each question
        in the category (using pol_mapping for individual politician answers and a default value if missing).
      - For each LLM (from model_answers), build its answer vector for those questions.
      - Compute the Pearson correlation coefficient between the party vector and the LLM vector.
      - Plot a heatmap of the correlation matrix, with rows corresponding to parties and columns to LLMs.

    Parameters:
      - category_map: dict mapping category ID to a dict with keys "name" and "question_ids"
      - party_to_pol_ids: dict mapping party short names to lists of politician IDs
      - pol_mapping: dict mapping politician IDs to their answer mapping (questionId -> answer)
      - model_answers: dict mapping model name to its answer mapping (questionId -> answer)
      - default: default answer value if a question is missing (default is 50.0)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Loop over each category.
    for cat_id, cat_info in category_map.items():
        qids = cat_info.get("question_ids", [])
        if not qids:
            continue  # Skip categories with no questions.

        category_name = cat_info.get("name", f"Category {cat_id}")
        parties = list(party_to_pol_ids.keys())
        models = list(model_answers.keys())

        # Prepare a matrix to store correlations (rows: parties, columns: models)
        corr_matrix = np.zeros((len(parties), len(models)))

        # For each party, compute its average answer vector for this category.
        for i, party in enumerate(parties):
            pol_ids = party_to_pol_ids.get(party, [])
            party_vector = []
            for qid in qids:
                values = []
                for pid in pol_ids:
                    mapping = pol_mapping.get(pid, {})
                    values.append(mapping.get(qid, default))
                # Average the answers for the current question across politicians.
                avg_val = np.mean(values) if values else default
                party_vector.append(avg_val)
            party_vector = np.array(party_vector)

            # For each model, build its answer vector for this category.
            for j, model in enumerate(models):
                model_mapping = model_answers.get(model, {})
                model_vector = np.array([model_mapping.get(qid, default) for qid in qids])

                # Compute the Pearson correlation coefficient if possible.
                if len(party_vector) > 1 and np.std(party_vector) > 0 and np.std(model_vector) > 0:
                    corr = np.corrcoef(party_vector, model_vector)[0, 1]
                else:
                    corr = 0
                corr_matrix[i, j] = corr

        # Plot the correlation matrix as a heatmap.
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")

        # Set tick labels.
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(parties)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticklabels(parties)
        ax.set_title(f"Correlation Matrix in {category_name}")

        # Annotate each cell with the correlation value.
        for i in range(len(parties)):
            for j in range(len(models)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=10)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


def main():
    """
    Minimal main function that loads data, processes it, and calls the plotting functions.
    (Uses load_json, load_questionnaire, load_category_map, build_answer_mapping,
    build_answer_vector, compute_agreement_percentage, and extract_party_info.)
    """
    # --- Data Loading and Preprocessing ---
    questionnaire_file = "./data/questionnaire/questionnaire.json"
    politicians_file   = "./data/answers/nationalrat_members.json"
    models_file        = "./data/answers/all_model_answers.json"

    # Load the list of all question IDs and the category mapping.
    question_ids = load_questionnaire(questionnaire_file)
    category_map = load_category_map(questionnaire_file)

    politicians = load_json(politicians_file)
    models      = load_models(models_file)

    model_selection = [
        "mistralai/mistral-large-2407", # Closed, European
        "mistralai/mixtral-8x22b-instruct", # Closed, European

        "qwen/qvq-72b-preview",
        "ai21/jamba-1-5-large",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",

        "amazon/nova-pro-v1",
        "google/gemini-2.0-flash-thinking-exp-1219:free",  # Closed, US
        "x-ai/grok-2-1212",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-405b-instruct",
        "openai/o1-preview"
    ]
    models = [m for m in models if m["name"] in model_selection]

    # Process politician answers.
    pol_vectors = {}   # key: politician id -> overall answer vector
    pol_party   = {}   # key: politician id -> party abbreviation
    for pol in politicians:
        pid = pol["id"]
        mapping = build_answer_mapping(pol.get("answers", []))
        vec = build_answer_vector(mapping, question_ids, default=50.0)
        pol_vectors[pid] = vec
        party_obj = pol.get("party", {})
        abbrev = party_obj.get("abbreviation", pol.get("partyAbbreviation"))
        pol_party[pid] = abbrev

    # Process LLM answers.
    # Build both a mapping (for spider charts) and overall vectors.
    model_answers = {}  # key: model name -> answer mapping (questionId -> value)
    model_vectors = {}  # key: model name -> overall answer vector
    for model in models:
        model_name = model["name"]
        answers = model.get("answers", [])
        if len(answers) == 0:
            continue
        mapping = build_answer_mapping(answers)
        model_answers[model_name] = mapping
        vec = build_answer_vector(mapping, question_ids, default=50.0)
        model_vectors[model_name] = vec

    # Process politician answers.
    # Build two dictionaries:
    #   - pol_vectors: politician id -> overall answer vector (for PCA, etc.)
    #   - pol_mapping: politician id -> answer mapping (for party spider charts)
    pol_vectors = {}
    pol_mapping = {}
    pol_party = {}
    for pol in politicians:
        pid = pol["id"]
        mapping = build_answer_mapping(pol.get("answers", []))
        pol_mapping[pid] = mapping
        vec = build_answer_vector(mapping, question_ids, default=50.0)
        pol_vectors[pid] = vec
        party_obj = pol.get("party", {})
        abbrev = party_obj.get("abbreviation", pol.get("partyAbbreviation"))
        pol_party[pid] = abbrev

    # Extract party info.
    party_info = extract_party_info(politicians)
    # Define a left-to-right ordering of parties.
    party_order = [
        "SP",
        "Grüne",
        # "BastA!",
        # "ALG",
        "GLP",
        "Die Mitte",
        "FDP",
        # "LDP",
        # "EVP",
        # "EDU",
        # "MCG",
        # "Lega",
        "SVP",
    ]
    party_order = [p for p in party_order if p in party_info]

    # Group politician IDs by party.
    party_to_pol_ids = {p: [] for p in party_order}
    for pid, party in pol_party.items():
        if party in party_to_pol_ids:
            party_to_pol_ids[party].append(pid)

    # Define marker styles for LLMs.
    markers = ["o", "s", "^", "D", "v", "p", "*", "1", "x", "<", ">", "P", "+"]
    model_marker = {}
    for i, model_name in enumerate(model_vectors.keys()):
        model_marker[model_name] = markers[i % len(markers)]

    # --- Create Plots ---
    # 1. PCA Plot.
    create_pca_plot(pol_vectors, model_vectors, pol_party, party_info, party_order, model_marker)

    # 2. Agreement Boxplots for each LLM.
    for model_name, m_vec in model_vectors.items():
        create_agreement_boxplot(model_name, m_vec, party_order, party_to_pol_ids, pol_vectors, party_info)

    # 3. Spider (Radar) Charts for each LLM.
    for model_name, llm_mapping in model_answers.items():
        create_spider_chart_for_llm(model_name, llm_mapping, category_map, default=50.0)

    for party in party_order:
        pol_ids = party_to_pol_ids.get(party, [])
        # Collect answer mappings for this party.
        party_pol_mappings = [pol_mapping[pid] for pid in pol_ids]
        # Compute the party's average answer mapping.
        party_avg_mapping = compute_party_average_mapping(party_pol_mappings, question_ids, default=50.0)
        # Get the party color from your party_info dictionary.
        party_color = party_info.get(party, {}).get("color", "gray")
        # Create the spider chart using the party's color.
        create_spider_chart_for_party(party, party_avg_mapping, category_map, party_color, default=50.0)

    create_1d_spectrum_plot(pol_vectors, pol_party, model_vectors, party_info, party_order, model_marker)

    create_correlation_matrix_by_category(category_map, party_to_pol_ids, pol_mapping, model_answers, default=50.0)


if __name__ == "__main__":
    main()