# -- Scrape the Knowledge Base, auto-login script

import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from requests_ntlm import HttpNtlmAuth
from urllib.parse import urljoin, urlparse, parse_qs
import urllib3

# ——— Disable insecure-request warnings ———
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ——— App Title ———
st.title("🔍 CCAH KB Scraper (with Skip-Logging)")

# ——— Credentials (must include DOMAIN\\username) ———
USERNAME = st.secrets["kb"]["username"]  
PASSWORD = st.secrets["kb"]["password"]

# ——— Configuration ———
BASE_URL        = "https://kb.ccah-alliance.org/"
LISTING_URL     = BASE_URL + "default.asp?id=403&Lang=1"
LISTING_WITH_SID= LISTING_URL + "&SID={sid}&page={page}"

BASE_SAVE       = "/home/shiraoka/projects/kb_agent/data/kbcontents"
TEXT_DIR        = os.path.join(BASE_SAVE, "text")
FILES_DIR       = os.path.join(BASE_SAVE, "files")
LOG_PATH        = os.path.join(BASE_SAVE, "skipped_files.log")

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

# ——— State ———
skipped_files = []

# ——— Helpers ———
def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name)

def truncate_url(full_url):
    p   = urlparse(full_url)
    qs  = parse_qs(p.query)
    pid = qs.get("id", [None])[0]
    if pid:
        return f"{p.scheme}://{p.netloc}{p.path}?id={pid}"
    return full_url

def download_file(session, file_url):
    if not file_url.lower().startswith(("http://", "https://")):
        skipped_files.append(file_url)
        st.warning(f"⚠️ Skipping unsupported URL: {file_url}")
        return
    fn   = os.path.basename(urlparse(file_url).path)
    dest = os.path.join(FILES_DIR, sanitize_filename(fn))
    try:
        r = session.get(file_url, stream=True, verify=False)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        st.write(f"✅ Downloaded file: {fn}")
    except Exception as e:
        skipped_files.append(file_url)
        st.error(f"❌ Failed to download {file_url}: {e}")

def extract_clean_text(soup, base_url):
    for tag in soup(["script", "style"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        raw       = a.get_text(strip=True)
        full_href = urljoin(base_url, a["href"])
        short_href= truncate_url(full_href)
        a.string  = f"{raw} ({short_href})"
    raw_text = soup.get_text(separator=" ", strip=True)
    return " ".join(raw_text.split())

def save_text(detail_url, soup):
    page_id = detail_url.split("id=")[-1].split("&")[0]
    fname   = f"page_{page_id}.txt"
    path    = os.path.join(TEXT_DIR, sanitize_filename(fname))
    text    = extract_clean_text(soup, detail_url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"URL: {truncate_url(detail_url)}\n\n")
        f.write(text)
    st.write(f"💾 Saved text: {fname}")

# ——— Scrape Logic ———
def scrape():
    # NTLM authentication on the listing page
    session = requests.Session()
    session.trust_env = False
    session.auth      = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # 1) Fetch landing page (NTLM handshake happens here)
    resp = session.get(LISTING_URL, verify=False)
    resp.raise_for_status()

    # 2) Extract real SID
    soup = BeautifulSoup(resp.text, "html.parser")
    first_link = soup.find("a", href=lambda h: h and "SID=" in h)
    if not first_link:
        st.error("❌ Unable to extract SID from landing page.")
        return
    sid = first_link["href"].split("SID=")[1].split("&")[0]
    st.write(f"🔑 Extracted SID: {sid}")

    # 3) Loop through the 7 listing pages
    for page in range(7):
        url = LISTING_WITH_SID.format(sid=sid, page=page)
        st.write(f"📄 Scraping listing page {page} → {url}")
        try:
            r = session.get(url, verify=False)
            r.raise_for_status()
        except Exception as e:
            st.error(f"❌ Unauthorized or error on page {page}: {e}")
            continue

        listing = BeautifulSoup(r.text, "html.parser")
        detail_links = [
            a["href"] for a in listing.find_all("a", href=True)
            if "default.asp?id=" in a["href"] and "&page=" not in a["href"]
        ]
        st.write(f"   → Found {len(detail_links)} detail links")

        # 4) Visit each detail page
        for href in detail_links:
            detail_url = urljoin(BASE_URL, href)
            st.write(f"     ↳ Fetching {detail_url}")
            dresp = session.get(detail_url, verify=False)
            dresp.raise_for_status()
            dsoup = BeautifulSoup(dresp.text, "html.parser")

            save_text(detail_url, dsoup)

            # 5) Download attachments
            for a in dsoup.find_all("a", href=True):
                f_url = urljoin(detail_url, a["href"])
                if f_url.lower().endswith((".pdf", ".doc", ".docx")):
                    download_file(session, f_url)

    # 6) Log skipped URLs
    if skipped_files:
        with open(LOG_PATH, "w", encoding="utf-8") as logf:
            logf.write("\n".join(skipped_files))
        st.success(f"✅ Scraping complete! Skipped {len(skipped_files)} URLs.")
        st.download_button(
            "📥 Download skipped-files log",
            data="\n".join(skipped_files),
            file_name="skipped_files.log"
        )
    else:
        st.success("✅ Scraping complete! No URLs were skipped.")

# ——— Trigger ———
if st.button("Start Scraping"):
    scrape()
