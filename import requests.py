import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
import time

# Load spaCy and add custom rules
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define Healthcare Policy Entities
patterns = [
    {"label": "GRANT", "pattern": [{"LOWER": "grant"}, {"LOWER": "awarded", "OP": "?"}]},
    {"label": "BUDGET", "pattern": [{"LOWER": "budget"}, {"LOWER": "allocation"}]},
    {"label": "WAIVER", "pattern": [{"TEXT": {"REGEX": "1115|1332"}}, {"LOWER": "waiver"}]},
    {"label": "MCO_CONTRACT", "pattern": [{"LOWER": "managed"}, {"LOWER": "care"}, {"LOWER": "contract"}]},
    {"label": "REIMBURSEMENT", "pattern": [{"LOWER": "reimbursement"}, {"LOWER": "rate"}]},
    {"label": "ENROLLMENT", "pattern": [{"LOWER": "enrollment"}, {"LOWER": "snapshot", "OP": "?"}]},
    {"label": "FEDERAL_MATCH", "pattern": [{"LOWER": "fmap"}, {"LOWER": "federal", "OP": "?"}, {"LOWER": "matching"}]}
]
ruler.add_patterns(patterns)

def analyze_article(text):
    doc = nlp(text)
    data = {
        "Grants": [], "Budgets": [], "Waivers": [], 
        "Contracts": [], "Rates": [], "Enrollment": [], "FMAP": []
    }
    
    # Logic to find the number NEAR the entity
    for ent in doc.ents:
        # Define window of context (10 tokens after the entity)
        context = doc[ent.start : ent.end + 10].text.lower()
        
        # Check for direction
        change = "increased" if "increase" in context or "award" in context else \
                 "decreased" if "decrease" in context or "cut" in context else "stable"
        
        # Extract the nearest number (Money, Percent, or Cardinal)
        vals = [e.text for e in doc.ents if e.start >= ent.start and e.start <= ent.end + 10 
                and e.label_ in ["MONEY", "PERCENT", "CARDINAL"]]
        val_str = f"{change}: {vals[0]}" if vals else f"{change}: N/A"

        if ent.label_ == "GRANT": data["Grants"].append(val_str)
        elif ent.label_ == "BUDGET": data["Budgets"].append(val_str)
        elif ent.label_ == "WAIVER": data["Waivers"].append(val_str)
        elif ent.label_ == "MCO_CONTRACT": data["Contracts"].append(val_str)
        elif ent.label_ == "REIMBURSEMENT": data["Rates"].append(val_str)
        elif ent.label_ == "ENROLLMENT": data["Enrollment"].append(val_str)
        elif ent.label_ == "FEDERAL_MATCH": data["FMAP"].append(val_str)

    return {k: "; ".join(set(v)) for k, v in data.items()}

# --- Mock Execution Flow ---
# (Using the scraper from previous steps to get 'df')
# df = scrape_dch_news(pages=2)
# analysis_results = df['text'].apply(analyze_article).apply(pd.Series)
# final_table = pd.concat([df[['date', 'title']], analysis_results], axis=1)