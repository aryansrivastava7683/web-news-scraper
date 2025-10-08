# 🌐 ML-Powered Global Impact Analyzer Chatbot

This project is a **chatbot-ready system** that allows users to input any query related to **commodities, companies, events, or natural disasters worldwide**, and outputs:

1. Recent headlines relevant to the query (scraped from multiple global news sources).  
2. Predicted **economic or event impact** using a **machine learning model**.

It uses a **TF-IDF + Logistic Regression** model trained on sample data and can be extended with more headlines or datasets.

---

## Features

- Scrapes live headlines from **RSS feeds** worldwide.  
- Detects **keywords** in user queries (commodities, companies, countries, events).  
- Predicts likely **impact** (positive, negative, neutral) using a **machine learning model**.  
- Chatbot-ready interface — interactive CLI.  
- Extensible: Add more commodities, companies, and rules or use semantic search for advanced matching.  

---

## Requirements

- Python 3.8+  
- Libraries:
  - `feedparser`
  - `scikit-learn`
  - *(Optional for future upgrades)* `nltk`, `numpy`, `pandas`, `sentence-transformers`

---

## Installation

1. **Clone the repository** (or copy the code into a folder):  
```bash
git clone https://github.com/yourusername/global-impact-chatbot.git
cd global-impact-chatbot
