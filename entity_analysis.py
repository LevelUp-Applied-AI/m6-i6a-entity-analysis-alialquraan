"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame unchanged
    return pd.read_csv(filepath)



def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    # TODO: Copy df, apply unicodedata.normalize('NFC', t) to each
    #       text, branch on language for English vs. Arabic handling,
    #       write results into a new `processed_text` column
    df_copy = df.copy()

    def normalize_text(row):
        text = row["text"]
        lang = row["language"]

        norm_text = unicodedata.normalize("NFC", text)

        if lang == "en":
            return norm_text
        elif lang == "ar":
            return norm_text  # أو ""
        else:
            return ""

    df_copy["processed_text"] = df_copy.apply(normalize_text, axis=1)
    return df_copy

def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to language == 'en', process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    en_df = df[df["language"] == "en"]

    records = []

    for _, row in en_df.iterrows():
        doc = nlp(row["text"]) 

        for ent in doc.ents:
            records.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(records)


def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count
                           (or filter to co_count >= 2) so the result
                           stays readable on the full corpus.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    # TODO: Count entity frequencies (top 20), compute label totals,
    #       build co-occurrence pairs, and join on articles_df.id to
    #       compute per-category entity-label counts
    top_entities = (
        entity_df
        .groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .head(20)
    )

    label_counts = entity_df["entity_label"].value_counts().to_dict()

    from itertools import combinations
    from collections import Counter

    co_counter = Counter()

    for text_id, group in entity_df.groupby("text_id"):
        entities = group["entity_text"].unique()

        for pair in combinations(sorted(entities), 2):
            co_counter[pair] += 1

    co_occurrence = pd.DataFrame([
        {"entity_a": k[0], "entity_b": k[1], "co_count": v}
        for k, v in co_counter.items()
    ])

    co_occurrence = co_occurrence.sort_values(by="co_count", ascending=False)

    co_occurrence = co_occurrence[co_occurrence["co_count"] >= 2].head(50)

    merged = entity_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    per_category = (
        merged
        .groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    print("\n--- Summary ---")
    print(f"Total entities: {len(entity_df)}")
    print(f"Unique entities: {entity_df['entity_text'].nunique()}")

    return {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    # TODO: Create a horizontal bar chart of top entities, colored or
    #       grouped by entity type, save to output_path
    top_entities = stats["top_entities"]

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_entities["entity_text"],
        top_entities["count"]
    )

    plt.xlabel("Frequency")
    plt.ylabel("Entity")
    plt.title("Top 20 Entities")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    # TODO: Build a formatted report string from the statistics
    label_counts = stats["label_counts"]
    top_entities = stats["top_entities"].head(5)

    top_pairs = co_occurrence.head(3) if co_occurrence is not None else None

    report = []

    report.append("=== Entity Analysis Report ===\n")

    report.append("Entity counts by type:\n")
    for label, count in label_counts.items():
        report.append(f"- {label}: {count}")

    report.append("\nTop 5 entities:\n")
    for _, row in top_entities.iterrows():
        report.append(f"- {row['entity_text']} ({row['entity_label']}): {row['count']}")

    report.append("\nTop co-occurring entity pairs:\n")
    if top_pairs is not None:
        for _, row in top_pairs.iterrows():
            report.append(f"- ({row['entity_a']}, {row['entity_b']}): {row['co_count']}")

    report.append("\nSummary:\n")
    report.append(
        "The corpus is dominated by a small set of high-frequency entities, "
        "primarily organizations, locations, and dates. Co-occurrence patterns "
        "suggest strong relationships between major institutions and key regions, "
        "reflecting structured discourse in climate-related topics."
    )

    return "\n".join(report)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
