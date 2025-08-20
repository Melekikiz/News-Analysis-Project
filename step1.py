import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# read the csv file
df = pd.read_csv("news.csv")

# text cleaning
df['text'] = df["title"].fillna('') + '. ' + df["description"].fillna('')

# Labels for zero-shot classification
labels = [
    "Politics", "Economy", "Technology", "Sports", "Health",
    "Science", "World", "Education", "Crime", "Entertainment"
]

# Keyword dictionary
keyword_dict = {
    "Politics": ["election", "president", "government", "minister", "politics", "policy", "law", "vote", "parliament", "democracy", "political"],
    "Economy": ["economy", "market", "finance", "stock", "business", "trade", "investment", "currency", "inflation"],
    "Technology": ["technology", "tech", "ai", "software", "hardware", "robots", "gadget", "device", "innovation", "startup", "app", "mobile", "internet"],
    "Sports": ["sports", "football", "tournament", "match", "league", "team", "player", "athlete", "game", "score", "win", "lose"],
    "Health": ["health", "pandemic", "disease", "hospital", "doctor", "diet", "weight", "nutrition", "food", "virus", "vaccine", "medicine", "surgery"],
    "Science": ["science", "nasa", "space", "research", "experiment", "discovery", "biology", "physics", "chemistry", "environment", "climate"],
    "World": ["world", "international", "global", "foreign", "united nations", "diplomacy", "conflict", "war", "peace", "human rights"],
    "Education": ["education", "school", "college", "university", "student", "teacher", "class", "course", "degree", "learning", "study", "exam"],
    "Crime": ["crime", "arrest", "police", "investigation"],
    "Entertainment": ["entertainment", "movie", "music", "celebrity", "tv", "show", "theater", "concert", "festival", "art", "culture", "drama"]
}

# Rule-based labeling
def rule_based_labels(text):
    found = []
    for cat, keywords in keyword_dict.items():
        if any(k in text for k in keywords):
            found.append(cat)
    return found

# Zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

predicted_labels = []
for text in tqdm(df['text'], desc="Labeling news"):
    if not isinstance(text, str) or text.strip() == "":
        predicted_labels.append("Unknown")
        continue

    text_lower = text.lower()

    # Rule-based
    rule_labels = rule_based_labels(text_lower)

    # Zero-shot multi-label
    result = classifier(
        text,
        labels,
        multi_label=True,
        hypothesis_template="This article is about {}."
    )

    zs_labels = [lbl for lbl, score in zip(result['labels'], result['scores']) if score > 0.4]

    # Merge rule-based + zero-shot
    final_labels = set(rule_labels + zs_labels)
    if not final_labels:
        final_labels = {"Unknown"}

    predicted_labels.append(", ".join(final_labels))

df['category'] = predicted_labels

# Source region mapping (bölge bazlı düzenledik)
source_region_map = {
    "The Guardian": "Europe",
    "Times of India": "Asia",
    "News-Medical": "North America",
    "Euronews.com": "Europe",
    "Associated Press of Pakistan": "Asia",
    "MindaNews": "Asia"
}

df['region'] = df['source'].map(source_region_map).fillna("Unknown")

# Save labeled news articles
df.to_csv("labeled_news.csv", index=False)
print("✅ Labeled news articles saved to 'labeled_news.csv'.")
