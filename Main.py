import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
import time
import re
import PyPDF2
from io import BytesIO

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
patterns += [
    {"label": "PROVIDER_COUNT", "pattern": [{"LOWER": "provider"}, {"LOWER": "count"}]},
    {"label": "ENROLLMENT_SNAPSHOT", "pattern": [{"LOWER": "enrollment"}, {"LOWER": "snapshot"}]},
    {"label": "WAIVER_APPROVAL", "pattern": [{"LOWER": "waiver"}, {"LOWER": "approval"}]},
    {"label": "MANAGED_CONTRACT", "pattern": [{"LOWER": "managed"}, {"LOWER": "contract"}]},
    {"label": "REIMBURSEMENT_RATE", "pattern": [{"LOWER": "reimbursement"}, {"LOWER": "rate"}]}
]
ruler.add_patterns(patterns)

def extract_text_from_pdf(pdf_url):
    """Download and extract text from a PDF URL"""
    try:
        print(f"      Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
            print(f"      PDF extracted: {len(text)} characters")
            return text
        else:
            print(f"      Failed to download PDF: status {response.status_code}")
            return ""
    except Exception as e:
        print(f"      Error extracting PDF: {e}")
        return ""

def clean_article_text(text):
    """Remove common boilerplate content from Georgia government websites"""
    boilerplate_phrases = [
        "An official website of the State of Georgia",
        "How you know",
        "The .gov means it's official",
        "Local, state, and federal government websites often end in .gov",
        "State of Georgia government websites and email systems use",
        "georgia.gov or ga.gov at the end of the address",
        "Secure websites use HTTPS certificates",
        "A lock icon or https:// means you've safely connected",
    ]
    
    cleaned = text
    for phrase in boilerplate_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def analyze_article(text):
    """Analyze article text and extract healthcare metrics"""
    doc = nlp(text)
    # Prepare output columns for each group
    groups = {
        'grants': [],
        'awarded': [],
        'budgets allocated': [],
        'federal matching': [],
        'provider counts': [],
        'enrollment snapshots': [],
        'waiver approvals': [],
        'managed contracts': [],
        'reimbursement rates': []
    }
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERCENT", "CARDINAL"]:
            start = max(ent.start - 10, 0)
            end = min(ent.end + 10, len(doc))
            context = doc[start:end].text.lower()
            # Assign to group based on context
            if "grant" in context:
                groups['grants'].append(ent.text)
            elif "award" in context or "awarded" in context:
                groups['awarded'].append(ent.text)
            elif "budget" in context or "allocation" in context or "allocated" in context:
                groups['budgets allocated'].append(ent.text)
            elif "fmap" in context or "federal match" in context or "federal matching" in context or "federal share" in context:
                groups['federal matching'].append(ent.text)
            elif "provider count" in context or "provider counts" in context or "provider" in context or "providers" in context:
                groups['provider counts'].append(ent.text)
            elif "enrollment snapshot" in context or "enrollment" in context or "enrollee" in context or "enrollees" in context:
                groups['enrollment snapshots'].append(ent.text)
            elif "waiver approval" in context or "waiver" in context or "approved" in context or "approval" in context:
                groups['waiver approvals'].append(ent.text)
            elif "managed contract" in context or "managed contracts" in context or "managed care" in context or "contract" in context:
                groups['managed contracts'].append(ent.text)
            elif "reimbursement rate" in context or "reimbursement" in context or "rate" in context:
                groups['reimbursement rates'].append(ent.text)
    # Join numbers for each group as a string
    return {k: ", ".join(set(v)) for k, v in groups.items()}

def scrape_article_content(article_url):
    """
    Scrape content from an article page, including any PDFs linked within the article
    Returns: combined text from HTML content and any PDFs found
    """
    try:
        article_resp = requests.get(article_url, timeout=10)
        if article_resp.status_code != 200:
            print(f"    Failed to fetch article: status {article_resp.status_code}")
            return ""
        
        article_soup = BeautifulSoup(article_resp.text, "html.parser")
        
        # Extract HTML content
        content_tag = (
            article_soup.select_one(".field--name-body") or
            article_soup.select_one(".field-name-body") or
            article_soup.select_one("article") or
            article_soup.select_one(".content")
        )
        
        html_text = ""
        if content_tag:
            html_text = content_tag.get_text(separator=" ", strip=True)
        else:
            paragraphs = article_soup.find_all('p')
            html_text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Look for PDF links in the article
        pdf_links = []
        for link in article_soup.find_all('a', href=True):
            href = link['href']
            # Check if link points to a PDF
            if href.endswith('.pdf') or 'pdf' in href.lower():
                # Make absolute URL if relative
                if href.startswith('http'):
                    pdf_url = href
                elif href.startswith('/'):
                    pdf_url = f"https://dch.georgia.gov{href}"
                else:
                    pdf_url = f"https://dch.georgia.gov/{href}"
                pdf_links.append(pdf_url)
        
        # Extract text from all PDFs found
        pdf_text = ""
        if pdf_links:
            print(f"    Found {len(pdf_links)} PDF link(s) in article")
            for pdf_url in pdf_links:
                pdf_content = extract_text_from_pdf(pdf_url)
                pdf_text += " " + pdf_content
        
        # Combine HTML and PDF text
        combined_text = html_text + " " + pdf_text
        return combined_text
        
    except Exception as e:
        print(f"    Error scraping article content: {e}")
        return ""

def scrape_dch_news(max_pages=19, delay=1.0):
    """
    Scrape articles from all pages of the DCH news site
    Args:
        max_pages: Maximum number of pages to scrape (default 19)
        delay: Seconds to wait between requests
    Returns:
        DataFrame with columns: date, title, text
    """
    base_url = "https://dch.georgia.gov/news?page="
    articles = []
    
    print(f"Starting scrape of {max_pages} pages...")
    
    for page in range(max_pages):
        print(f"\nScraping page {page + 1}/{max_pages}...")
        url = f"{base_url}{page}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"  Page {page} returned status {resp.status_code}, stopping.")
                break
        except Exception as e:
            print(f"  Error fetching page {page}: {e}")
            break
            
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all article links on this page
        article_links = []
        for a in soup.find_all('a', href=True):
            m = re.match(r"/announcement/(\d{4}-\d{2}-\d{2})/(.+)", a['href'])
            if m:
                date = m.group(1)
                title = a.get_text(strip=True)
                link = a['href']
                article_url = f"https://dch.georgia.gov{link}"
                article_links.append((date, title, article_url))
        
        print(f"  Found {len(article_links)} article links on page {page + 1}")
        
        # Process each article on this page
        for idx, (date, title, article_url) in enumerate(article_links, 1):
            print(f"  Processing article {idx}/{len(article_links)}: {title[:60]}...")
            
            # Scrape article content (includes PDF extraction if found)
            text = scrape_article_content(article_url)
            
            articles.append({
                "date": date,
                "title": title,
                "text": text,
                "url": article_url
            })
            
            # Small delay between articles
            time.sleep(0.5)
        
        # Delay between pages
        time.sleep(delay)
    
    # Remove duplicates
    df = pd.DataFrame(articles).drop_duplicates(subset=["date", "title"])
    return df.reset_index(drop=True)

if __name__ == "__main__":
    print("="*70)
    print("DCH News Scraper - Full 19 Page Scrape with PDF Support")
    print("="*70)
    
    # Scrape all 19 pages
    df = scrape_dch_news(max_pages=19, delay=1.0)
    print(f"\n{'='*70}")
    print(f"Total articles scraped: {len(df)}")
    print(f"{'='*70}\n")
    
    if df.empty or 'text' not in df.columns:
        print("ERROR: No articles found. Check scraping logic or website structure.")
    else:
        print("Analyzing all articles for healthcare metrics...")
        analysis_results = df['text'].apply(analyze_article).apply(pd.Series)
        final_table = pd.concat([df[['date', 'title']], analysis_results], axis=1)
        
        # Display summary
        print("\n" + "="*70)
        print("Data Summary")
        print("="*70)
        for col in ["grants", "awarded", "budgets allocated", "federal matching", 
                    "provider counts", "enrollment snapshots", "waiver approvals", 
                    "managed contracts", "reimbursement rates"]:
            count = (final_table[col] != "").sum()
            print(f"{col:.<30} {count} articles with data")
        
        # Save to CSV
        output_file = 'dch_news_analysis_complete.csv'
        final_table.to_csv(output_file, index=False)
        print(f"\n{'='*70}")
        print(f"Full results saved to '{output_file}'")
        print(f"{'='*70}\n")
        
        # Display preview of results
        display_table = final_table.copy()
        display_table['title'] = display_table['title'].str.slice(0, 50)
        print("\nPreview - First 10 rows:")
        print(display_table.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("Scraping complete!")
        print("="*70)