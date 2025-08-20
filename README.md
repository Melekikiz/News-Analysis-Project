# News-Analysis-Project
This project demonstrates a full pipeline for analyzing news articles, starting from raw, unlabeled data to visual insights about categories, regions, word trends, and more.

Project Overview
We collected 900+ news articles from multiple sources using APIs. Initially, these articles were untagged, containing only titles, descriptions, sources, and timestamps.
The project pipeline consists of three main steps:

1️⃣ Data Cleaning & Preprocessing

Combined article titles and descriptions into a single text field.
Removed special characters, lowercased text, and filtered out stopwords.
Extracted publication hour for temporal analysis.

2️⃣ Automatic Labeling

Applied rule-based keyword labeling using a custom dictionary.
Applied zero-shot classification with HuggingFace's facebook/bart-large-mnli to detect multiple categories per article.
Merged rule-based and model-based predictions for robust multi-label assignments.
Mapped article sources to regions (Europe, Asia, North America, etc.).

3️⃣ Visualization & Insights

Category Distribution: See which categories are most common across all articles.
Region Distribution: Compare category prevalence across different regions.

Word Trends:

Most frequent words overall.
Top TF-IDF words (both global and per region).
Word cloud to visualize the content distribution.
Temporal Analysis: Explore when news articles are published during the day.

Dashboards

news_step1_dashboard.png: Main category, region, hourly, and TF-IDF insights.
news_step2_dashboard.png: Word-focused dashboard, including top words, region-specific heatmaps, and distribution pie chart.
news_wordcloud.png: Word cloud of all news articles.

Key Takeaways

Combining rule-based methods and zero-shot learning allows us to label previously untagged data effectively.
Visualizations reveal global trends, such as which topics dominate certain regions or which words are most frequent across all news.
TF-IDF and region-specific heatmaps help identify important keywords for each region.

Tools & Libraries

Python
pandas, matplotlib, seaborn, wordcloud
sklearn (CountVectorizer, TfidfVectorizer)
transformers (HuggingFace zero-shot classification)
tqdm for progress bars

Project completed by **Melek Ikiz**
