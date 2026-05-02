"""
Module 6 Week A — Stretch: Multilingual NER Comparison

Run multilingual NER using spaCy and Hugging Face on
English and Arabic texts, then compare results.

Run: python stretch_multilingual_ner.py
"""

import unicodedata
import pandas as pd
import spacy
from transformers import pipeline


def load_corpus(filepath="data/climate_articles.csv"):
    """Load dataset"""
    return pd.read_csv(filepath)


def preprocess_corpus(df):
    df_copy = df.copy()

    def normalize_text(text):
        return unicodedata.normalize("NFC", text)

    df_copy["processed_text"] = df_copy["text"].apply(normalize_text)
    return df_copy


def split_languages(df, n=20):
    en = df[df["language"] == "en"].head(n)
    ar = df[df["language"] == "ar"].head(n)
    return en, ar


def run_spacy_ner(df, nlp):
    results = []

    for _, row in df.iterrows():
        doc = nlp(row["processed_text"])

        for ent in doc.ents:
            results.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_
            })

    return pd.DataFrame(results)


def run_hf_ner(df, hf_model):
    results = []

    for _, row in df.iterrows():
        entities = hf_model(row["processed_text"])

        for ent in entities:
            results.append({
                "text_id": row["id"],
                "entity_text": ent["word"],
                "entity_label": ent["entity"]
            })

    return pd.DataFrame(results)


def compute_stats(entity_df, df):

    if len(entity_df) == 0:
        return {
            "total": 0,
            "label_counts": {},
            "density": 0,
            "examples": []
        }

    total = len(entity_df)
    label_counts = entity_df["entity_label"].value_counts().to_dict()

    total_words = df["processed_text"].str.split().apply(len).sum()
    density = (total / total_words) * 100 if total_words > 0 else 0

    examples = entity_df.head(3).to_dict("records")

    return {
        "total": total,
        "label_counts": label_counts,
        "density": density,
        "examples": examples
    }


def print_comparison(results):

    print("\n=== Multilingual NER Comparison ===\n")

    for key, value in results.items():
        lang, model = key

        print(f"--- {lang.upper()} | {model} ---")
        print(f"Total Entities: {value['total']}")
        print(f"Density: {value['density']:.2f}")

        print("Label Counts:")
        for k, v in value["label_counts"].items():
            print(f"  {k}: {v}")

        print("Examples:")
        for ex in value["examples"]:
            print(f"  {ex['entity_text']} ({ex['entity_label']})")

        print()


if __name__ == "__main__":

    print("Loading models...")

    nlp_spacy = spacy.load("xx_ent_wiki_sm")

    hf_ner = pipeline("ner", model="Davlan/xlm-roberta-base-wikiann-ner")

    print("Loading dataset...")
    df = load_corpus()

    df = preprocess_corpus(df)

    en_df, ar_df = split_languages(df)

    print(f"English texts: {len(en_df)}")
    print(f"Arabic texts: {len(ar_df)}")

    results = {}

    spacy_en = run_spacy_ner(en_df, nlp_spacy)
    hf_en = run_hf_ner(en_df, hf_ner)

    spacy_ar = run_spacy_ner(ar_df, nlp_spacy)
    hf_ar = run_hf_ner(ar_df, hf_ner)

    results[("english", "spacy")] = compute_stats(spacy_en, en_df)
    results[("english", "hf")] = compute_stats(hf_en, en_df)
    results[("arabic", "spacy")] = compute_stats(spacy_ar, ar_df)
    results[("arabic", "hf")] = compute_stats(hf_ar, ar_df)

    print_comparison(results)