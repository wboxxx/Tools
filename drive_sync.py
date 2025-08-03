import json
import logging
from pathlib import Path
from typing import Dict, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import yaml

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DEFAULT_ROOT = Path("/data/google_mirror")


class DriveMirrorer:
    """Mirror a shared Google Drive folder locally."""

    def __init__(self, folder_id: str, root: Path | str = DEFAULT_ROOT, *, verbose: bool = False):
        self.folder_id = folder_id
        self.root = Path(root)
        self.files_root = self.root / "files"
        self.shards_root = self.root / "shards"
        self.index_file = self.root / "index.json"
        self.files_root.mkdir(parents=True, exist_ok=True)
        self.shards_root.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Dict] = self._load_index()
        self.creds = self._load_credentials()
        self.service = build("drive", "v3", credentials=self.creds)
        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.info("Initialised DriveMirrorer for folder %s", self.folder_id)

    # ------------------------------------------------------------------
    def _load_credentials(self) -> Credentials:
        if Path("token.json").exists():
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w", encoding="utf-8") as token:
                token.write(creds.to_json())
        return creds

    def _load_index(self) -> Dict[str, Dict]:
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def sync(self) -> None:
        """Synchronise new files from Drive."""
        try:
            logger.info("Starting sync for folder %s", self.folder_id)
            query = f"'{self.folder_id}' in parents and trashed=false"
            response = self.service.files().list(q=query, fields="files(id, name, mimeType, size, modifiedTime)").execute()
            files = response.get("files", [])
            logger.debug("Fetched %d file(s) from Drive", len(files))
            downloaded = 0
            for file_meta in files:
                if file_meta["id"] in self.index:
                    logger.debug("Skipping existing file %s", file_meta["name"])
                    continue
                logger.info("Downloading %s", file_meta["name"])
                self._download_file(file_meta)
                downloaded += 1
            logger.info("Sync complete: %d new file(s) downloaded", downloaded)
        except HttpError as exc:
            logger.error("Drive sync failed: %s", exc)

    def _download_file(self, meta: Dict) -> None:
        file_id = meta["id"]
        local_path = self.files_root / meta["name"]
        logger.debug("Requesting download of %s to %s", file_id, local_path)
        request = self.service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug("Download %s progress: %d%%", file_id, int(status.progress() * 100))
        logger.debug("Download finished for %s", file_id)
        text = extract_text(local_path)
        logger.debug("Extracted %d characters from %s", len(text), local_path)
        self.index[file_id] = {
            "name": meta.get("name"),
            "path": str(local_path),
            "size": int(meta.get("size", 0)),
            "modified": meta.get("modifiedTime"),
            "mime": meta.get("mimeType"),
            "text": text,
        }
        self._save_index()
        generate_shard(local_path, self.index[file_id], self.shards_root)


# ----------------------------------------------------------------------
def extract_text(path: Path) -> str:
    logger.debug("Extracting text from %s", path)
    try:
        if path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".pdf":
            from pdfminer.high_level import extract_text
            return extract_text(str(path))
        if path.suffix.lower() == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("text extraction failed for %s: %s", path, exc)
    return ""


def run_llama_pipeline(text: str) -> Dict:
    """Very small summarisation stub using transformers."""
    try:
        from transformers import pipeline
        summariser = pipeline("summarization", model="hf-internal-testing/tiny-random-t5")
        summary = summariser(text[:1000], max_length=60, truncation=True)[0]["summary_text"]
        return {"summary": summary, "tags": []}
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("summarisation failed: %s", exc)
        return {"summary": "", "tags": []}


def generate_shard(src: Path, meta: Dict, shards_root: Path) -> None:
    shards_root.mkdir(exist_ok=True)
    logger.debug("Generating shard for %s", src)
    nlp = run_llama_pipeline(meta.get("text", ""))
    shard = {
        "source": str(src),
        "summary": nlp["summary"],
        "tags": nlp["tags"],
    }
    shard_path = shards_root / f"{src.stem}.yaml"
    with open(shard_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(shard, f, allow_unicode=True)


def search_shards(query: str, shards_root: Path | str = DEFAULT_ROOT / "shards") -> List[Dict]:
    root = Path(shards_root)
    results: List[Dict] = []
    for path in root.glob("*.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        haystacks = [data.get("summary", "")] + data.get("tags", [])
        if any(query.lower() in h.lower() for h in haystacks):
            data["_shard_path"] = str(path)
            results.append(data)
    return results


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Google Drive mirroring tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    sub = parser.add_subparsers(dest="cmd")

    sync_p = sub.add_parser("sync", help="Synchronise new files")
    sync_p.add_argument("folder_id", help="Shared folder ID")

    query_p = sub.add_parser("query", help="Search in shards")
    query_p.add_argument("text", help="Query text")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.cmd == "sync":
        DriveMirrorer(args.folder_id, verbose=args.verbose).sync()
    elif args.cmd == "query":
        for item in search_shards(args.text):
            print(f"{item['source']}: {item['summary']}")


if __name__ == "__main__":  # pragma: no cover - CLI usage
    main()
