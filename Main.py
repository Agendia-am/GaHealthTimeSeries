import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import tempfile
import io
import os
import re
import time
import spacy
from spacy.pipeline import EntityRuler
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Session with automatic retries (free reliability upgrade)
# ---------------------------------------------------------------------------

def create_session(retries: int = 3, backoff: float = 0.5,
                   timeout: int = 15) -> requests.Session:
    """Build a requests.Session with retry / back-off strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "DCHNewsParser/1.0 (research; educational)"
    })
    session._default_timeout = timeout
    return session


# ---------------------------------------------------------------------------
# Wayback Machine API  (free – archive.org)
# ---------------------------------------------------------------------------

WAYBACK_API = "https://archive.org/wayback/available"


def wayback_lookup(url: str, session: requests.Session) -> Optional[str]:
    """Query the Wayback Machine API for a cached snapshot of *url*."""
    try:
        resp = session.get(WAYBACK_API, params={"url": url},
                           timeout=session._default_timeout)
        data = resp.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available"):
            return snapshot["url"]
    except Exception:
        pass
    return None


def fetch_html(url: str, session: requests.Session) -> tuple[str, str]:
    """Fetch HTML for *url*.  Falls back to Wayback Machine API on failure.
    Returns (html_text, source) where source is 'live' or 'wayback'."""
    try:
        resp = session.get(url, timeout=session._default_timeout)
        resp.raise_for_status()
        return resp.text, "live"
    except requests.RequestException:
        print(f"    [API] Live fetch failed, trying Wayback Machine API …")
        wb_url = wayback_lookup(url, session)
        if wb_url:
            resp = session.get(wb_url, timeout=session._default_timeout)
            resp.raise_for_status()
            print(f"    [API] ✓ Got cached version from Wayback Machine")
            return resp.text, "wayback"
        raise


# ---------------------------------------------------------------------------
# Boilerplate removal
# ---------------------------------------------------------------------------

BOILERPLATE = [
    "An official website of the State of Georgia",
    "How you know",
    "The .gov means it's official",
    "Local, state, and federal government websites often end in .gov",
    "State of Georgia government websites and email systems use",
    "georgia.gov or ga.gov at the end of the address",
    "Secure websites use HTTPS certificates",
    "A lock icon or https:// means you've safely connected",
]


def clean_text(raw: str) -> str:
    """Strip Georgia.gov boilerplate phrases and collapse whitespace."""
    text = raw
    for phrase in BOILERPLATE:
        text = text.replace(phrase, "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# PDF parsing via PyPDFLoader + RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_pdf_data(pdf_url: str, session: requests.Session):
    """Download PDF via PyPDFLoader, return (full_text, chunks).
    Falls back to Wayback Machine API if the direct download fails."""
    full_text = ""
    chunks: list[str] = []
    tmp_path = None
    try:
        # Try live download first, fall back to Wayback API
        try:
            resp = session.get(pdf_url, timeout=session._default_timeout)
            resp.raise_for_status()
        except requests.RequestException:
            wb_url = wayback_lookup(pdf_url, session)
            if wb_url:
                resp = session.get(wb_url, timeout=session._default_timeout)
                resp.raise_for_status()
            else:
                return "", []

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        for doc in documents:
            full_text += doc.page_content + "\n"

        split_docs = text_splitter.split_documents(documents)
        chunks = [d.page_content for d in split_docs]

    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return full_text.strip(), chunks


# ---------------------------------------------------------------------------
# NLP Setup: spaCy Healthcare Metric Categories
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "GRANT", "pattern": [{"LOWER": "grant"}]},
    {"label": "BUDGET", "pattern": [{"LOWER": "budget"}, {"LOWER": "allocation"}]},
    {"label": "WAIVER", "pattern": [{"TEXT": {"REGEX": "1115|1332"}}, {"LOWER": "waiver"}]},
    {"label": "FEDERAL_MATCH", "pattern": [{"LOWER": "fmap"}, {"LOWER": "federal"}, {"LOWER": "matching"}]},
    {"label": "ENROLLMENT", "pattern": [{"LOWER": "enrollment"}, {"LOWER": "snapshot"}]},
    {"label": "PROVIDER", "pattern": [{"LOWER": "provider"}, {"LOWER": "count"}]},
    {"label": "REIMBURSEMENT", "pattern": [{"LOWER": "reimbursement"}]}
]
ruler.add_patterns(patterns)


# ---------------------------------------------------------------------------
# Free Local Embeddings  (sentence-transformers / all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the free local embedding model (~80 MB, cached)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[embeddings] Loading free model '{EMBEDDING_MODEL_NAME}' …")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate 384-dim embeddings for a list of texts. Free, runs locally."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, batch_size=64,
                           show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ArticleRecord:
    date: str
    title: str
    url: str
    body: str = ""
    pdf_text: str = ""
    source: str = "live"           # "live" or "wayback" (from API)
    chunks: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)
    # Healthcare metric columns (from spaCy NLP)
    grants: str = ""
    awarded: str = ""
    budgets_allocated: str = ""
    federal_matching: str = ""
    provider_counts: str = ""
    enrollment_snapshots: str = ""
    waiver_approvals: str = ""
    managed_contracts: str = ""
    reimbursement_rates: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Extraction & Analysis
# ---------------------------------------------------------------------------

def analyze_for_metrics(text):
    """Extracts numeric values related to healthcare categories from text."""
    doc = nlp(text)
    data = {
        "grants": [], "awarded": [], "budgets_allocated": [],
        "federal_matching": [], "provider_counts": [],
        "enrollment_snapshots": [], "waiver_approvals": [],
        "managed_contracts": [], "reimbursement_rates": []
    }

    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERCENT", "CARDINAL"]:
            context = doc[max(ent.start - 10, 0) : min(ent.end + 10, len(doc))].text.lower()

            if "grant" in context: data["grants"].append(ent.text)
            if "award" in context: data["awarded"].append(ent.text)
            if "budget" in context or "allocat" in context: data["budgets_allocated"].append(ent.text)
            if "fmap" in context or "federal match" in context: data["federal_matching"].append(ent.text)
            if "provider" in context: data["provider_counts"].append(ent.text)
            if "enroll" in context: data["enrollment_snapshots"].append(ent.text)
            if "waiver" in context: data["waiver_approvals"].append(ent.text)
            if "managed" in context or "contract" in context: data["managed_contracts"].append(ent.text)
            if "reimbursement" in context: data["reimbursement_rates"].append(ent.text)

    return {k: ", ".join(set(v)) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

def scrape_dch_to_csv(max_pages=1):
    session = create_session()
    base_url = "https://dch.georgia.gov/news?page="
    all_records = []
    all_chunks_for_embedding = []  # (record_index, chunk_text)

    for page in range(max_pages):
        print(f"\n[parser] Page {page + 1}/{max_pages} …")

        # Fetch index page via session (with Wayback API fallback)
        try:
            html, src = fetch_html(f"{base_url}{page}", session)
        except Exception as exc:
            print(f"  Could not fetch page {page}: {exc}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        seen = set()

        for a in soup.find_all("a", href=True):
            match = re.search(r"/announcement/(\d{4}-\d{2}-\d{2})/(.+)", a['href'])
            if not match or a['href'] in seen:
                continue
            seen.add(a['href'])

            url = "https://dch.georgia.gov" + a['href']
            record = ArticleRecord(
                date=match.group(1),
                title=a.get_text(strip=True),
                url=url,
                source=src,
            )
            print(f"  [parser] {record.title[:60]} …")

            # 1. Fetch article (with Wayback API fallback)
            try:
                art_html, art_src = fetch_html(url, session)
                record.source = art_src
                art_soup = BeautifulSoup(art_html, "html.parser")

                # Extract body with boilerplate removal
                content = (
                    art_soup.select_one(".field--name-body") or
                    art_soup.select_one(".field-name-body") or
                    art_soup.select_one("article") or
                    art_soup.select_one(".content")
                )
                if content:
                    record.body = clean_text(content.get_text(separator=" ", strip=True))
                else:
                    record.body = clean_text(art_soup.get_text(separator=" ", strip=True))

                # 2. Find and extract PDFs (PyPDFLoader + Wayback fallback)
                for link in art_soup.find_all("a", href=True):
                    href = link['href']
                    if href.lower().endswith(".pdf") or "/files/" in href.lower():
                        pdf_url = href if href.startswith("http") else "https://dch.georgia.gov" + href
                        pdf_txt, pdf_chunks = extract_pdf_data(pdf_url, session)
                        record.pdf_text += (" " + pdf_txt)
                        record.chunks.extend(pdf_chunks)

                record.pdf_text = record.pdf_text.strip()

                # Split article body into chunks too
                if record.body:
                    body_chunks = text_splitter.split_text(record.body)
                    record.chunks = body_chunks + record.chunks

            except Exception as exc:
                print(f"    Error: {exc}")

            # 3. Analyze text for healthcare metrics (spaCy NLP)
            combined_text = record.body + " " + record.pdf_text
            metrics = analyze_for_metrics(combined_text)
            for key, val in metrics.items():
                setattr(record, key, val)

            # Track chunks for batch embedding
            rec_idx = len(all_records)
            for chunk in record.chunks:
                all_chunks_for_embedding.append((rec_idx, chunk))

            all_records.append(record)
            time.sleep(0.5)

    # 4. Generate free local embeddings for all chunks
    if all_chunks_for_embedding:
        print(f"\n[embeddings] Embedding {len(all_chunks_for_embedding)} chunks "
              f"with '{EMBEDDING_MODEL_NAME}' (free, local) …")
        chunk_texts = [c[1] for c in all_chunks_for_embedding]
        vectors = embed_texts(chunk_texts)
        for (rec_idx, _), vec in zip(all_chunks_for_embedding, vectors):
            all_records[rec_idx].embeddings.append(vec)
        print(f"[embeddings] ✓ {len(vectors)} embeddings generated (dim={len(vectors[0])})")

    # Build DataFrame and save
    df = pd.DataFrame([r.to_dict() for r in all_records])
    if not df.empty:
        df.drop_duplicates(subset=["date", "title"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ── Export CSV matching the target format ──────────────────────────
    csv_columns = [
        "date", "title", "grants", "awarded",
        "budgets_allocated", "federal_matching",
        "provider_counts", "enrollment_snapshots",
        "waiver_approvals", "managed_contracts",
        "reimbursement_rates",
    ]
    csv_df = df[csv_columns].copy()
    csv_df.columns = [
        "date", "title", "grants", "awarded",
        "budgets allocated", "federal matching",
        "provider counts", "enrollment snapshots",
        "waiver approvals", "managed contracts",
        "reimbursement rates",
    ]

    desktop_path = os.path.expanduser("~/Desktop/dch_news_analysis.csv")
    csv_df.to_csv(desktop_path, index=False)
    csv_df.to_csv("dch_news_analysis.csv", index=False)
    df.to_json("dch_news_analysis.json", orient="records", indent=2)

    print(f"\n{'=' * 70}")
    print(f"Total records : {len(csv_df)}")
    print(f"APIs used     : Wayback Machine (fallback), sentence-transformers (embeddings)")
    print(f"{'=' * 70}")

    if not csv_df.empty:
        preview = csv_df.copy()
        preview["title"] = preview["title"].str[:55]
        print(preview.head(10).to_string(index=False))

    print(f"\n✓ Saved  {desktop_path}")
    print("✓ Saved  dch_news_analysis.csv")
    print("✓ Saved  dch_news_analysis.json")
    return df


if __name__ == "__main__":
    scrape_dch_to_csv(max_pages=10)