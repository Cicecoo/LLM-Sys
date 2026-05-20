from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import argparse
import hashlib
import mimetypes
import re
import shutil
import subprocess

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://llmsystem.github.io/llmsystemhomework/assignment_3/"
OUT = Path("llmsys_hw3/README.md")
ASSETS = Path("llmsys_hw3/assets/assignment_3")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def curl_get(url: str, insecure: bool) -> bytes:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl/curl.exe not found")

    cmd = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        "60",
        "--retry",
        "4",
        "--retry-delay",
        "1",
        "--user-agent",
        USER_AGENT,
    ]
    if insecure:
        cmd.append("--insecure")
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or f"curl failed with exit code {result.returncode}")
    return result.stdout


def get_bytes(session: requests.Session, url: str, insecure: bool) -> tuple[bytes, str]:
    try:
        response = session.get(url, timeout=30, verify=not insecure)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "")
    except requests.RequestException as requests_error:
        try:
            return curl_get(url, insecure), ""
        except RuntimeError as curl_error:
            raise RuntimeError(
                f"Failed to download {url}\n"
                f"requests error: {requests_error}\n"
                f"curl error: {curl_error}"
            ) from requests_error


parser = argparse.ArgumentParser(
    description="Download the Assignment 3 page body as a GitHub-friendly README.md."
)
parser.add_argument(
    "--insecure",
    action="store_true",
    help="Disable TLS certificate verification. Use only if your network/proxy breaks HTTPS.",
)
args = parser.parse_args()

session = build_session()
html = get_bytes(session, URL, args.insecure)[0]
soup = BeautifulSoup(html, "html.parser")
article = soup.select_one("article.md-content__inner")
if article is None:
    raise RuntimeError("Cannot find article.md-content__inner")

for tag in article.select("a.headerlink, .md-content__button, script, style"):
    tag.decompose()

ASSETS.mkdir(parents=True, exist_ok=True)
used_names = set()

for img in article.find_all("img"):
    src = img.get("src")
    if not src:
        continue
    img_url = urljoin(URL, src)
    content, content_type = get_bytes(session, img_url, args.insecure)

    name = Path(unquote(urlparse(img_url).path)).name
    if "." not in name:
        ctype = content_type.split(";")[0]
        name += mimetypes.guess_extension(ctype) or ".png"

    target = ASSETS / name
    if target.name in used_names:
        h = hashlib.sha1(img_url.encode()).hexdigest()[:8]
        target = ASSETS / f"{target.stem}-{h}{target.suffix}"
    used_names.add(target.name)

    target.write_bytes(content)
    img["src"] = f"assets/assignment_3/{target.name}"

for a in article.find_all("a", href=True):
    href = a["href"]
    if not href.startswith("#") and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href):
        a["href"] = urljoin(URL, href)

text = md(str(article), heading_style="ATX", bullets="-").strip()
text = re.sub(r"\n{3,}", "\n\n", text)

OUT.write_text(text + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
