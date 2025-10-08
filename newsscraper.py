import feedparser
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from datetime import datetime


rss_feeds = [
    "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html"
]


commodities = [
    "gold", "silver", "crude oil", "natural gas", "coal", "wheat", "rice", "coffee", "copper", "fuel", "gasoline"
]
companies = [
    "Tesla", "Apple", "Microsoft", "Google", "Amazon", "Reliance", "Tata", "ONGC", "Adani", "Infosys", "Meta",
    "Netflix", "Nvidia", "Toyota", "Samsung", "Shell", "Exxon", "BP"
]
countries = [
    "India", "United States", "China", "Japan", "Germany", "France", "UK", "Italy", "Canada", "Australia",
    "Brazil", "Russia", "South Korea", "South Africa", "Mexico", "Saudi Arabia", "Indonesia"
]


train_texts = [
    "crude oil prices surge affecting airline companies negatively",
    "gold prices rise as markets fall",
    "technology companies see growth after dollar weakens",
    "earthquake in japan disrupts semiconductor production",
    "heavy rain in assam impacts agriculture sector",
    "flood in gujarat increases infrastructure spending",
    "natural gas price increase hurts manufacturing sector",
    "stock market rallies after interest rate cut",
    "falling oil prices benefit transport industry",
    "rise in steel demand boosts mining companies"
]
train_labels = [
    "negative", "negative", "positive", "negative", "negative", "positive", "negative", "positive", "positive", "positive"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])

model.fit(train_texts, train_labels)


def scrape_headlines():
    headlines = []
    for url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                headlines.append({
                    "title": entry.title,
                    "summary": getattr(entry, "summary", ""),
                    "published": getattr(entry, "published", "Unknown"),
                    "link": entry.link
                })
        except Exception:
            continue
    return headlines


def extract_keywords(query):
    return re.findall(r'\w+', query.lower())


def search_and_predict(headlines, query):
    keywords = extract_keywords(query)
    results = []
    for h in headlines:
        combined = (h["title"] + " " + h["summary"]).lower()
        if any(k in combined for k in keywords):
            prediction = model.predict([combined])[0]
            results.append({
                "headline": h["title"],
                "impact": prediction,
                "link": h["link"],
                "published": h["published"]
            })
    return results


def global_impact_chatbot(query):
    print(f"\n Analyzing query: {query}")
    headlines = scrape_headlines()
    matches = search_and_predict(headlines, query)

    if not matches:
        return f"No related headlines found for '{query}'."

    response = f"\n Predicted Global Impacts for '{query}':\n"
    for m in matches[:5]:
        response += f"- {m['headline']} | Impact: {m['impact']} ({m['published']})\n  🔗 {m['link']}\n"
    return response


if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your global impact query (e.g., 'gold price in india impact on telecom'):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            break
        print(global_impact_chatbot(user_query))
