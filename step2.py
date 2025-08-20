import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud


df = pd.read_csv("labeled_news.csv")
df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
df['hour'] = pd.to_datetime(df['publishedAt']).dt.hour

# Region Mapping
source_region_map = {
    "The Guardian": "Europe",
    "Times of India": "Asia",
    "News-Medical": "North America",
    "Euronews.com": "Europe",
    "Associated Press of Pakistan": "Asia",
    "MindaNews": "Asia"
}
df['region'] = df['source'].map(source_region_map).fillna("Unknown")


# TF-IDF (category-specific words)

tfidf = TfidfVectorizer(max_features=15, stop_words="english")
X_tfidf = tfidf.fit_transform(df['text'].astype(str))
tfidf_scores = X_tfidf.toarray().sum(axis=0)
tfidf_df = pd.DataFrame({"word": tfidf.get_feature_names_out(), "tfidf": tfidf_scores}).sort_values(by="tfidf", ascending=False)


# Dashboard

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Category Distribution
sns.countplot(y="category", data=df, order=df['category'].value_counts().index, palette="viridis", ax=axes[0,0])
axes[0,0].set_title("Category Distribution", fontsize=14)

# Region Distribution
sns.countplot(y="region", data=df, order=df['region'].value_counts().index, palette="viridis", ax=axes[0,1])
axes[0,1].set_title("Region Distribution", fontsize=14)

# Hourly Distribution (per region)
sns.histplot(data=df, x="hour", hue="region", multiple="stack", palette="viridis", bins=24, ax=axes[1,0])
axes[1,0].set_title("News by Hour & Region", fontsize=14)

# TF-IDF Top Words
sns.barplot(x="tfidf", y="word", data=tfidf_df.head(15), palette="viridis", ax=axes[1,1])
axes[1,1].set_title("Top Words by TF-IDF", fontsize=14)

plt.tight_layout()
plt.savefig("news_step1_dashboard.png", dpi=150)
plt.show()




# Common Words (overall)

vectorizer = CountVectorizer(max_features=20, stop_words="english")
X = vectorizer.fit_transform(df['text'].astype(str))
word_counts = X.toarray().sum(axis=0)
word_df = pd.DataFrame({"word": vectorizer.get_feature_names_out(), "count": word_counts})


# Region-specific word frequencies

region_vectorizer = CountVectorizer(max_features=10, stop_words="english")
region_word_counts = {}
for region in df['region'].unique():
    texts = df[df['region'] == region]['text'].astype(str)
    X_region = region_vectorizer.fit_transform(texts)
    counts = X_region.toarray().sum(axis=0)
    region_word_counts[region] = dict(zip(region_vectorizer.get_feature_names_out(), counts))
region_word_df = pd.DataFrame(region_word_counts).fillna(0)


# TF-IDF Global

tfidf = TfidfVectorizer(max_features=20, stop_words="english")
X_tfidf = tfidf.fit_transform(df['text'].astype(str))
tfidf_scores = X_tfidf.toarray().sum(axis=0)
tfidf_df = pd.DataFrame({"word": tfidf.get_feature_names_out(), "tfidf": tfidf_scores}).sort_values(by="tfidf", ascending=False)


# Dashboard

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Overall Top Words
sns.barplot(x="count", y="word", data=word_df.sort_values("count", ascending=False).head(15), palette="magma", ax=axes[0,0])
axes[0,0].set_title("Most Common Words (Overall)", fontsize=14)

#TF-IDF Words
sns.barplot(x="tfidf", y="word", data=tfidf_df.head(15), palette="magma", ax=axes[0,1])
axes[0,1].set_title("Top Words by TF-IDF (Global)", fontsize=14)

#Region-Specific Heatmap
sns.heatmap(region_word_df, cmap="magma", annot=True, fmt=".0f", ax=axes[1,0])
axes[1,0].set_title("Top Words by Region", fontsize=14)

#Word Distribution (Pie)
axes[1,1].pie(word_df["count"].head(10), labels=word_df["word"].head(10), autopct="%1.1f%%", colors=sns.color_palette("magma", 10))
axes[1,1].set_title("Top 10 Words Distribution", fontsize=14)

plt.tight_layout()
plt.savefig("news_step2_dashboard.png", dpi=150)
plt.show()



text_all = " ".join(df['text'].astype(str))
wc = WordCloud(width=1000, height=600, background_color="white", colormap="viridis").generate(text_all)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of News Text", fontsize=16)
plt.savefig("news_wordcloud.png", dpi=150)
plt.show()
