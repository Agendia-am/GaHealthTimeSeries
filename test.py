"""
DCH News Parser
================
A structured parser for Georgia Department of Community Health news articles.
Uses requests + BeautifulSoup to fetch pages, then parses HTML into structured
records. Includes Wayback Machine API fallback, retry logic, PDF table extraction,
and data validation.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import tempfile
import os
import re
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.evaluation import load_evaluator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
import streamlit as st


# ---------------------------------------------------------------------------
# Data model – every parsed article is stored as an ArticleRecord
# ---------------------------------------------------------------------------

@dataclass
class ArticleRecord:
    """Structured representation of a single parsed DCH news article."""
    date: str
    title: str
    url: str
    body: str = ""
    pdf_urls: list = field(default_factory=list)
    pdf_text: str = ""
    pdf_pages: list = field(default_factory=list)   # per-page text from PyPDFLoader
    chunks: list = field(default_factory=list)      # text splitter output
    embeddings: list = field(default_factory=list)   # ada-002 embeddings per chunk
    source: str = "live"          # "live" or "wayback"
    fetched_at: str = ""
    valid: bool = True
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Session factory with automatic retries
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
    # store timeout so callers can use it
    session._default_timeout = timeout
    return session


# ---------------------------------------------------------------------------
# Wayback Machine API helpers
# ---------------------------------------------------------------------------

WAYBACK_API = "https://archive.org/wayback/available"


def wayback_lookup(url: str, session: requests.Session) -> Optional[str]:
    """Query the Wayback Machine API for a cached snapshot of *url*.
    Returns the snapshot URL or None."""
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
    """Fetch HTML for *url*.  Falls back to Wayback Machine on failure.
    Returns (html_text, source) where source is 'live' or 'wayback'."""
    try:
        resp = session.get(url, timeout=session._default_timeout)
        resp.raise_for_status()
        return resp.text, "live"
    except requests.RequestException:
        wb_url = wayback_lookup(url, session)
        if wb_url:
            resp = session.get(wb_url, timeout=session._default_timeout)
            resp.raise_for_status()
            return resp.text, "wayback"
        raise


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

# CSS selectors tried in order when locating the article body
BODY_SELECTORS = [
    ".field--name-body",
    ".field-name-body",
    "article .content",
    "article",
    ".node__content",
    ".content",
]

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
    """Strip boilerplate phrases and collapse whitespace."""
    text = raw
    for phrase in BOILERPLATE:
        text = text.replace(phrase, "")
    return re.sub(r"\s+", " ", text).strip()


def parse_article_body(soup: BeautifulSoup) -> str:
    """Extract the main body text from an article page."""
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            return clean_text(node.get_text(separator=" ", strip=True))
    # fallback – concatenate all <p>
    return clean_text(
        " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    )


def parse_pdf_links(soup: BeautifulSoup, base: str = "https://dch.georgia.gov") -> list[str]:
    """Return absolute URLs for every PDF linked in the page."""
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") or "/files/" in href.lower():
            if href.startswith("http"):
                urls.append(href)
            elif href.startswith("/"):
                urls.append(f"{base}{href}")
            else:
                urls.append(f"{base}/{href}")
    return list(dict.fromkeys(urls))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# PDF parsing  (PyPDFLoader + RecursiveCharacterTextSplitter)
# ---------------------------------------------------------------------------

# Shared text splitter – tune chunk_size / overlap to your needs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_pdf_text_and_chunks(pdf_url: str, session: requests.Session):
    """Download a PDF via PyPDFLoader, split into chunks.

    Returns
    -------
    full_text : str
        Concatenated text from all pages.
    pages : list[dict]
        Per-page text + metadata from PyPDFLoader.
    chunks : list[str]
        Text chunks produced by RecursiveCharacterTextSplitter.
    """
    full_text = ""
    pages: list[dict] = []
    chunks: list[str] = []
    tmp_path = None
    try:
        # Download PDF to a temp file (PyPDFLoader needs a file path)
        resp = session.get(pdf_url, timeout=session._default_timeout)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Load with PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()  # list[Document]

        for doc in documents:
            pages.append({
                "page": doc.metadata.get("page", 0),
                "text": doc.page_content,
            })
            full_text += doc.page_content + "\n"

        # Split the loaded documents into chunks
        split_docs = text_splitter.split_documents(documents)
        chunks = [d.page_content for d in split_docs]

    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return full_text.strip(), pages, chunks


# ---------------------------------------------------------------------------
# Index page parser – turns the news listing into ArticleRecord stubs
# ---------------------------------------------------------------------------

ARTICLE_RE = re.compile(r"/announcement/(\d{4}-\d{2}-\d{2})/(.+)")


def parse_index_page(html: str, source: str = "live") -> list[ArticleRecord]:
    """Parse a single DCH /news?page=N listing page into ArticleRecords."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    records: list[ArticleRecord] = []
    for a in soup.find_all("a", href=True):
        m = ARTICLE_RE.match(a["href"])
        if not m:
            continue
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        records.append(ArticleRecord(
            date=m.group(1),
            title=a.get_text(strip=True),
            url=f"https://dch.georgia.gov{href}",
            source=source,
            fetched_at=datetime.now(tz=timezone.utc).isoformat(),
        ))
    return records


# ---------------------------------------------------------------------------
# Full article parser
# ---------------------------------------------------------------------------

def parse_article(record: ArticleRecord, session: requests.Session) -> ArticleRecord:
    """Fetch and parse the full content of a single article in-place."""
    try:
        html, source = fetch_html(record.url, session)
        record.source = source
        soup = BeautifulSoup(html, "html.parser")
        record.body = parse_article_body(soup)
        record.pdf_urls = parse_pdf_links(soup)

        # Parse each linked PDF with PyPDFLoader + text splitter
        for pdf_url in record.pdf_urls:
            pdf_txt, pdf_pages, pdf_chunks = extract_pdf_text_and_chunks(
                pdf_url, session
            )
            if pdf_txt:
                record.pdf_text += (" " + pdf_txt)
            record.pdf_pages.extend(pdf_pages)
            record.chunks.extend(pdf_chunks)

        record.pdf_text = record.pdf_text.strip()

        # Also split the article body into chunks
        if record.body:
            body_chunks = text_splitter.split_text(record.body)
            record.chunks = body_chunks + record.chunks
    except Exception as exc:
        record.valid = False
        record.errors.append(str(exc))
    return record


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(record: ArticleRecord) -> ArticleRecord:
    """Check that required fields are present and well-formed."""
    if not re.match(r"\d{4}-\d{2}-\d{2}", record.date):
        record.valid = False
        record.errors.append(f"Invalid date format: {record.date}")
    if not record.title:
        record.valid = False
        record.errors.append("Missing title")
    if not record.body and not record.pdf_text:
        record.valid = False
        record.errors.append("No body text or PDF text extracted")
    return record


# ---------------------------------------------------------------------------
# Free Local Embeddings  (sentence-transformers / all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

# Model is downloaded once (~80 MB) then cached locally.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Lazy-load the sentence-transformers model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[embeddings] Loading model '{model_name}' (first run downloads ~80 MB) …")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def embed_chunks(
    records: list[ArticleRecord],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 64,
) -> list[ArticleRecord]:
    """Generate embeddings for every chunk in each record.

    Uses the free, local sentence-transformers model (all-MiniLM-L6-v2).
    No API key required — runs entirely on your machine.

    Parameters
    ----------
    records : list[ArticleRecord]
        Parsed records whose `.chunks` will be embedded.
    model_name : str
        HuggingFace model name (default: all-MiniLM-L6-v2, 384-dim).
    batch_size : int
        Chunks encoded per batch.

    Returns
    -------
    The same list of records, with `.embeddings` populated.
    """
    # Collect all chunks across all records into one flat list
    chunk_index: list[tuple[int, int]] = []  # (record_idx, chunk_idx)
    all_texts: list[str] = []
    for r_idx, rec in enumerate(records):
        for c_idx, chunk in enumerate(rec.chunks):
            chunk_index.append((r_idx, c_idx))
            all_texts.append(chunk)

    if not all_texts:
        print("[embeddings] No chunks to embed.")
        return records

    model = get_embedding_model(model_name)
    print(f"[embeddings] Embedding {len(all_texts)} chunks with '{model_name}' …")

    # Encode in batches (returns numpy arrays)
    vectors = model.encode(all_texts, batch_size=batch_size,
                           show_progress_bar=False, normalize_embeddings=True)

    # Map vectors back to their records
    for (r_idx, _), vector in zip(chunk_index, vectors):
        records[r_idx].embeddings.append(vector.tolist())

    dim = len(vectors[0])
    print(f"[embeddings] Done – {len(vectors)} embeddings generated (dim={dim}).")
    return records


# ---------------------------------------------------------------------------
# Top-level parser driver
# ---------------------------------------------------------------------------

def parse_dch_news(max_pages: int = 1, delay: float = 1.0,
                   parse_full_articles: bool = True) -> pd.DataFrame:
    """
    Parse the Georgia DCH news site and return a structured DataFrame.

    Parameters
    ----------
    max_pages : int
        Number of listing pages to process (default 1).
    delay : float
        Seconds to wait between HTTP requests.
    parse_full_articles : bool
        If True, follow each link and parse the full article + PDFs.

    Returns
    -------
    pd.DataFrame  with columns:
        date, title, url, body, pdf_urls, pdf_text, chunks, embeddings,
        source, fetched_at, valid, errors
    """
    session = create_session()
    all_records: list[ArticleRecord] = []

    for page_num in range(max_pages):
        page_url = f"https://dch.georgia.gov/news?page={page_num}"
        print(f"[parser] Fetching index page {page_num + 1}/{max_pages} …")

        try:
            html, src = fetch_html(page_url, session)
        except Exception as exc:
            print(f"[parser] Could not fetch page {page_num}: {exc}")
            continue

        stubs = parse_index_page(html, source=src)
        print(f"[parser]   → {len(stubs)} articles found")

        if parse_full_articles:
            for i, rec in enumerate(stubs, 1):
                print(f"[parser]   Parsing article {i}/{len(stubs)}: "
                      f"{rec.title[:60]} …")
                parse_article(rec, session)
                validate_record(rec)
                time.sleep(delay * 0.3)

        all_records.extend(stubs)
        time.sleep(delay)

    # Generate embeddings for all parsed chunks (free, local model)
    if all_records:
        try:
            embed_chunks(all_records)
        except Exception as exc:
            print(f"[embeddings] Skipped – {exc}")

    df = pd.DataFrame([r.to_dict() for r in all_records])
    if not df.empty:
        df.drop_duplicates(subset=["date", "title"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCH News Parser")
    parser.add_argument("--pages", type=int, default=1,
                        help="Number of listing pages to parse (default 1)")
    args = parser.parse_args()

    print("=" * 70)
    print("DCH News Parser  –  Parsing + Embeddings (all-MiniLM-L6-v2, free)")
    print("=" * 70)

    df = parse_dch_news(
        max_pages=args.pages,
        delay=1.0,
        parse_full_articles=True,
    )

    print(f"\n{'=' * 70}")
    print(f"Total parsed records : {len(df)}")
    print(f"Valid records        : {df['valid'].sum() if not df.empty else 0}")
    print(f"{'=' * 70}\n")

    if not df.empty:
        # Preview
        df["num_chunks"] = df["chunks"].apply(len)
        df["num_embeddings"] = df["embeddings"].apply(len)
        df["num_pdf_pages"] = df["pdf_pages"].apply(len)
        preview = df[["date", "title", "source", "valid",
                      "num_chunks", "num_embeddings", "num_pdf_pages"]].copy()
        preview["title"] = preview["title"].str[:45]
        print(preview.to_string(index=False))

        # Save structured output as CSV (Desktop + local) and JSON
        desktop_path = os.path.expanduser("~/Desktop/dch_parsed_output.csv")
        df.to_csv(desktop_path, index=False)
        df.to_csv("dch_parsed_output.csv", index=False)
        df.to_json("dch_parsed_output.json", orient="records", indent=2)
        print(f"\n✓ Saved  {desktop_path}")
        print("✓ Saved  dch_parsed_output.csv")
        print("✓ Saved  dch_parsed_output.json")

        # Print first 5 rows
        print(f"\n{'=' * 70}")
        print("First 5 rows:")
        print("=" * 70)
        print(df.head(5).to_string())
    else:
        print("No records parsed.")