# NEXUS Commerce Intelligence

> End-to-end AI & ML platform for e-commerce intelligence — scraping, data analysis, machine learning and MLOps.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scrapy](https://img.shields.io/badge/Scrapy-2.14-green)
![MLflow](https://img.shields.io/badge/MLflow-3.10-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## What it does

NEXUS scrapes product data from e-commerce sites, cleans and analyzes it, trains ML models to extract insights, and serves everything through an interactive dashboard — all with experiment tracking and MLOps best practices.

## Live demo

> Dashboard coming soon on Streamlit Community Cloud

## Project structure
```
nexus-commerce/
├── scraping/          # Scrapy spiders and data pipelines
├── data/
│   ├── raw/           # Raw scraped data (JSON)
│   └── processed/     # Cleaned datasets (CSV)
├── models/            # Trained ML models + MLflow experiments
├── dashboard/         # Streamlit interactive dashboard
├── notebooks/         # Data cleaning and exploration scripts
└── requirements.txt   # All dependencies
```

## Stack

| Layer | Technology |
|-------|-----------|
| Scraping | Scrapy 2.14, Playwright, BeautifulSoup4 |
| Data | pandas, pymongo, MongoDB Atlas |
| ML | scikit-learn, XGBoost, Random Forest |
| MLOps | MLflow 3.10, experiment tracking, model registry |
| Dashboard | Streamlit, Plotly |
| Scheduling | APScheduler |
| Vector DB | ChromaDB |
| CI/CD | GitHub Actions |

## Results — Books dataset

- 1,000 products scraped across 50 pages in 61 seconds
- 983 items/minute throughput
- Baseline ML model: Random Forest rating predictor (19.5% accuracy — price as single feature)
- Next step: NLP features from title text to improve prediction

## How to run locally
```bash
# Clone the repo
git clone https://github.com/koichi055/nexus-commerce.git
cd nexus-commerce

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run the scraper
cd scraping
scrapy crawl books -o ../data/raw/books.json

# Run the dashboard
cd ../dashboard
streamlit run app.py

# Open MLflow UI
cd ..
mlflow ui --backend-store-uri models/mlruns
```

## Roadmap

- [x] Scrapy spider — Books to Scrape (1,000 products)
- [x] Data cleaning pipeline with pandas
- [x] Baseline ML model with MLflow tracking
- [x] Interactive Streamlit dashboard
- [ ] Mercado Libre spider (real e-commerce data)
- [ ] NLP pipeline — sentiment analysis on reviews
- [ ] Dynamic pricing model with reinforcement learning
- [ ] Recommendation engine (Two-Tower neural network)
- [ ] Deploy dashboard on Streamlit Community Cloud

## Author

**Koichi Rodriguez** — AI & ML Engineer | Data Scientist  
[GitHub](https://github.com/koichi055) · [LinkedIn](#) · [Upwork](#)