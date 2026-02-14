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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from spacy.pipeline import EntityRuler
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import  , timezone

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


def fetch_html(url: str, session: requests.Session) -> tuple[str, str] | tuple[None, None]:
    """Fetch HTML for *url*.  Falls back to Wayback Machine API on failure.
    Returns (html_text, source) where source is 'live' or 'wayback'.
    Returns (None, None) if both fail — caller should skip this URL."""
    try:
        resp = session.get(url, timeout=session._default_timeout)
        resp.raise_for_status()
        return resp.text, "live"
    except requests.RequestException:
        print(f"    [API] Live fetch failed, trying Wayback Machine API …")
        try:
            wb_url = wayback_lookup(url, session)
            if wb_url:
                resp = session.get(wb_url, timeout=session._default_timeout)
                resp.raise_for_status()
                print(f"    [API] ✓ Got cached version from Wayback Machine")
                return resp.text, "wayback"
        except Exception:
            pass
        print(f"    [API] ✗ Both live and Wayback failed — skipping {url[:70]}")
        return None, None


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
# Recursive Deep Crawl – follow every internal DCH link
# ---------------------------------------------------------------------------

# Thread-safe visited set shared across all parallel workers
_visited_lock = threading.Lock()


def _mark_visited(url: str, visited: set) -> bool:
    """Thread-safely check-and-add *url* to *visited*.  Returns True if new."""
    normalised = url.split("?")[0].split("#")[0].rstrip("/")
    with _visited_lock:
        if normalised in visited:
            return False
        visited.add(normalised)
        return True


def deep_crawl_dch(
    start_url: str,
    session: requests.Session,
    *,
    max_depth: int = 3,
    visited: set | None = None,
    _depth: int = 0,
) -> tuple[str, str, list[str]]:
    """Crawl *start_url*, following only links **inside the article body**.

    This avoids spidering the entire site via nav / sidebar / footer links.
    PDFs found anywhere on the page are extracted immediately.
    A shared *visited* set (thread-safe) prevents duplicate fetches.

    Returns (body_text, pdf_text, chunks).
    """
    if visited is None:
        visited = set()

    if not _mark_visited(start_url, visited) or _depth > max_depth:
        return "", "", []

    body_text = ""
    pdf_text = ""
    chunks: list[str] = []

    html, _src = fetch_html(start_url, session)
    if html is None:
        return "", "", []

    soup = BeautifulSoup(html, "html.parser")

    # --- locate the article content area ---
    content_area = (
        soup.select_one(".field--name-body") or
        soup.select_one(".field-name-body") or
        soup.select_one("article") or
        soup.select_one(".content")
    )
    if content_area:
        body_text = clean_text(content_area.get_text(separator=" ", strip=True))
    else:
        body_text = clean_text(soup.get_text(separator=" ", strip=True))
        content_area = soup  # fallback: scan entire page

    if _depth > 0:
        tag = "sub-page" if _depth == 1 else f"depth-{_depth}"
        print(f"      [{tag}] {start_url[:80]}")

    # --- only follow links INSIDE the content area (not nav/footer) ---
    child_links: list[str] = []
    for a_tag in content_area.find_all("a", href=True):
        href = a_tag["href"]

        # Build absolute URL
        if href.startswith("/"):
            href = "https://dch.georgia.gov" + href
        elif not href.startswith("http"):
            continue  # skip mailto:, javascript:, etc.

        href_clean = href.split("?")[0].split("#")[0].rstrip("/")

        # PDF → extract immediately
        if href.lower().endswith(".pdf") or "/files/" in href.lower():
            if _mark_visited(href, visited):
                print(f"      [pdf] {href[:80]}")
                ptxt, pchunks = extract_pdf_data(href, session)
                pdf_text += " " + ptxt
                chunks.extend(pchunks)
        # Internal DCH page → queue for recursive crawl
        elif "dch.georgia.gov" in href_clean:
            if not re.search(r"\.(css|js|jpg|png|gif|svg|ico|xml|rss)$", href, re.I):
                child_links.append(href)

    # --- recurse into child DCH content pages ---
    for child_url in child_links:
        c_body, c_pdf, c_chunks = deep_crawl_dch(
            child_url, session,
            max_depth=max_depth, visited=visited, _depth=_depth + 1,
        )
        body_text += " " + c_body
        pdf_text += " " + c_pdf
        chunks.extend(c_chunks)

    return body_text.strip(), pdf_text.strip(), chunks


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

# Keywords for each category — expanded for maximum capture
_CATEGORY_KEYWORDS = {
    "grants":              ["grant", "funded", "funding", "mini-grant", "award grant"],
    "awarded":             ["award", "won", "received", "selected", "approv", "boost",
                            "invest", "infus", "total"],
    "budgets_allocated":   ["budget", "allocat", "appropriat", "spend", "expenditure",
                            "outlay", "fiscal", "sfy", "fy"],
    "federal_matching":    ["fmap", "federal match", "federal share", "match rate",
                            "federal fund", "cms approv", "medicaid fund"],
    "provider_counts":     ["provider", "hospital", "facilit", "clinic", "physician",
                            "nurse", "consultant", "lactation"],
    "enrollment_snapshots":["enroll", "member", "beneficiar", "recipient", "particip",
                            "eligible", "covered", "insured", "uninsured"],
    "waiver_approvals":    ["waiver", "1115", "1332", "1135", "1915", "appendix k",
                            "demonstration", "renewal", "amend"],
    "managed_contracts":   ["contract", "managed", "cmo", "procurement", "vendor",
                            "rfp", "solicitation", "bid", "proposal", "d-snp",
                            "care management", "benefit manager", "pbm", "pace"],
    "reimbursement_rates": ["reimburs", "rate", "payment", "copay", "co-pay",
                            "premium", "cost", "fee", "per diem", "directed pay"],
}


def analyze_for_metrics(text: str) -> dict[str, str]:
    """Extract EVERY number from *text* and sort into healthcare categories.

    • Uses spaCy NER (MONEY, PERCENT, CARDINAL, QUANTITY, ORDINAL, DATE).
    • Also runs regex to catch dollar amounts and percentages spaCy misses.
    • Every number is placed in every category whose keywords appear in the
      surrounding context.  Numbers that match NO category still go into
      the nearest plausible column so nothing is lost.
    """
    doc = nlp(text)
    data: dict[str, list[str]] = {k: [] for k in _CATEGORY_KEYWORDS}

    # --- 1. spaCy named entities that look like numbers ---
    number_labels = {"MONEY", "PERCENT", "CARDINAL", "QUANTITY", "ORDINAL"}
    seen_spans: set[tuple[int, int]] = set()          # avoid double-counting

    for ent in doc.ents:
        if ent.label_ not in number_labels:
            continue
        span_key = (ent.start, ent.end)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        context = doc[max(ent.start - 15, 0): min(ent.end + 15, len(doc))].text.lower()
        matched = False
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in context for kw in keywords):
                data[cat].append(ent.text)
                matched = True
        if not matched:
            # Put uncategorized money in 'awarded', others in closest fit
            if ent.label_ == "MONEY":
                data["awarded"].append(ent.text)
            elif ent.label_ == "PERCENT":
                data["reimbursement_rates"].append(ent.text)
            else:
                data["enrollment_snapshots"].append(ent.text)

    # --- 2. Regex fallback: catch $X / X% patterns spaCy may miss ---
    for m in re.finditer(
        r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|[MBKmk])?"
        r"|[\d,]+(?:\.\d+)?\s*%"
        r"|[\d,]+(?:\.\d+)?\s+(?:million|billion|thousand)",
        text,
    ):
        value = m.group().strip()
        # Check if spaCy already caught this span
        char_start, char_end = m.start(), m.end()
        already = False
        for ent in doc.ents:
            if ent.start_char <= char_start and ent.end_char >= char_end:
                already = True
                break
        if already:
            continue

        # Grab surrounding context (±80 chars)
        ctx_start = max(char_start - 80, 0)
        ctx_end = min(char_end + 80, len(text))
        context = text[ctx_start:ctx_end].lower()

        matched = False
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in context for kw in keywords):
                data[cat].append(value)
                matched = True
        if not matched:
            if "$" in value:
                data["awarded"].append(value)
            elif "%" in value:
                data["reimbursement_rates"].append(value)
            else:
                data["enrollment_snapshots"].append(value)

    # Deduplicate while preserving order
    return {k: ", ".join(dict.fromkeys(v)) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------

# Number of parallel workers for article scraping
MAX_WORKERS = 6


def _process_article(
    date: str, title: str, url: str, source: str,
    session: requests.Session, global_visited: set,
) -> ArticleRecord:
    """Process a single article: deep-crawl + NLP.  Runs in a worker thread."""
    record = ArticleRecord(date=date, title=title, url=url, source=source)
    try:
        crawl_body, crawl_pdf, crawl_chunks = deep_crawl_dch(
            url, session, max_depth=3, visited=global_visited,
        )
        record.body = crawl_body
        record.pdf_text = crawl_pdf
        record.chunks = crawl_chunks

        if record.body:
            body_chunks = text_splitter.split_text(record.body)
            record.chunks = body_chunks + record.chunks

    except Exception as exc:
        print(f"    Error ({title[:40]}): {exc}")

    # spaCy NLP metric extraction
    combined_text = record.body + " " + record.pdf_text
    metrics = analyze_for_metrics(combined_text)
    for key, val in metrics.items():
        setattr(record, key, val)

    return record


def scrape_dch_to_csv(max_pages=1):
    session = create_session()
    base_url = "https://dch.georgia.gov/news?page="
    all_records = []
    all_chunks_for_embedding = []  # (record_index, chunk_text)
    global_visited: set = set()     # shared across all workers (thread-safe)

    # --- Phase 1: collect article URLs from all listing pages ---
    article_jobs: list[tuple[str, str, str, str]] = []  # (date, title, url, src)
    seen_hrefs: set = set()

    for page in range(max_pages):
        print(f"\n[parser] Scanning page {page + 1}/{max_pages} …")
        html, src = fetch_html(f"{base_url}{page}", session)
        if html is None:
            print(f"  Could not fetch page {page} — skipping")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            match = re.search(r"/announcement/(\d{4}-\d{2}-\d{2})/(.+)", a['href'])
            if not match or a['href'] in seen_hrefs:
                continue
            seen_hrefs.add(a['href'])
            url = "https://dch.georgia.gov" + a['href']
            article_jobs.append((match.group(1), a.get_text(strip=True), url, src))

    print(f"\n[parallel] Scraping {len(article_jobs)} articles with {MAX_WORKERS} workers …")

    # --- Phase 2: deep-crawl articles in parallel ---
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for date, title, url, src in article_jobs:
            fut = pool.submit(
                _process_article, date, title, url, src, session, global_visited,
            )
            futures[fut] = title

        for i, fut in enumerate(as_completed(futures), 1):
            title_short = futures[fut][:55]
            try:
                record = fut.result()
                all_records.append(record)
                print(f"  [{i}/{len(futures)}] ✓ {title_short}")
            except Exception as exc:
                print(f"  [{i}/{len(futures)}] ✗ {title_short}: {exc}")

    # Sort records by date (descending) to match the target CSV ordering
    all_records.sort(key=lambda r: r.date, reverse=True)

    # Collect chunks for batch embedding
    for rec_idx, record in enumerate(all_records):
        for chunk in record.chunks:
            all_chunks_for_embedding.append((rec_idx, chunk))

    # 3. Generate free local embeddings for all chunks
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
    scrape_dch_to_csv(max_pages=1)