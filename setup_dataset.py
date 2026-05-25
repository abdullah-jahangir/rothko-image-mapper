"""
setup_dataset.py
Download Rothko paintings from WikiArt and build color profiles.

USAGE
-----
Download everything + build profiles:
    python setup_dataset.py

After you've manually deleted images you don't want, rebuild profiles:
    python setup_dataset.py --reindex

Fetch descriptions, gallery names, tags etc for existing profiles:
    python setup_dataset.py --enrich
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils.color_extractor import extract_palette

# ── Config ────────────────────────────────────────────────────────────────────
WIKIART_URL = (
    "https://www.wikiart.org/en/App/Painting/PaintingsByArtist"
    "?artistUrl=mark-rothko&json=2"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
DOWNLOAD_DELAY = 0.4  # seconds between requests (be polite to WikiArt)
TIMEOUT = 15

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
PROFILES_PATH = DATA_DIR / "profiles.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(url: str) -> str:
    """Turn an image URL into a safe local filename."""
    name = url.split("/")[-1].split("!")[0]
    safe = "".join(c for c in name if c.isalnum() or c in "-_.")
    return safe or "painting.jpg"


def fetch_painting_list() -> list[dict]:
    print("Fetching full Rothko catalogue from WikiArt…")
    resp = requests.get(WIKIART_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        paintings = data
    elif isinstance(data, dict):
        paintings = data.get("Paintings", data.get("paintings", []))
    else:
        paintings = []
    # Keep only entries that actually have an image URL
    paintings = [p for p in paintings if p.get("image")]
    print(f"  → {len(paintings)} paintings with image URLs")
    return paintings


# ── Mode 1: download everything ───────────────────────────────────────────────
def download_all(paintings: list[dict]) -> list[dict]:
    """Download every painting and extract its colour profile."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    profiles = []

    for i, painting in enumerate(paintings, 1):
        title = painting.get("title", "Untitled")
        year  = painting.get("completitionYear", "?")
        image_url  = painting.get("image", "")
        content_id = str(painting.get("contentId", i))

        filename = slugify(image_url)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            filename += ".jpg"
        dest = IMG_DIR / filename

        print(f"[{i}/{len(paintings)}] {title} ({year})")

        # Download (skip if already on disk)
        try:
            if dest.exists():
                print(f"  ✓ Already downloaded: {filename}")
            else:
                r = requests.get(image_url, headers=HEADERS, timeout=TIMEOUT)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"  ↓ {len(r.content) // 1024} KB → {filename}")
                time.sleep(DOWNLOAD_DELAY)
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            continue

        # Colour profile
        try:
            palette = extract_palette(dest, n_clusters=6)
            profiles.append({
                "id":        content_id,
                "title":     title,
                "year":      year,
                "filename":  filename,
                "image_url": image_url,
                "palette":   palette,
            })
            print(f"  🎨 Dominant colour: RGB {palette[0]['rgb']}")
        except Exception as e:
            print(f"  ✗ Palette extraction failed: {e}")
            if dest.exists():
                dest.unlink()  # remove broken file so it retries next run

    return profiles


# ── Mode 2: reindex from whatever images remain ───────────────────────────────
def reindex(paintings: list[dict]) -> list[dict]:
    """
    Rebuild profiles.json from the images currently in data/images/.
    Use this after you've manually deleted paintings you don't want.
    """
    # Build a lookup: filename → painting metadata (for title/year)
    meta: dict[str, dict] = {}
    for p in paintings:
        fname = slugify(p.get("image", ""))
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            fname += ".jpg"
        meta[fname] = p

    existing = sorted(IMG_DIR.glob("*"))
    image_files = [
        f for f in existing
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print(f"Found {len(image_files)} images in {IMG_DIR}")
    profiles = []

    for i, img_path in enumerate(image_files, 1):
        fname = img_path.name
        painting = meta.get(fname, {})
        title = painting.get("title", fname)
        year  = painting.get("completitionYear", "?")
        content_id = str(painting.get("contentId", fname))
        image_url  = painting.get("image", "")

        print(f"[{i}/{len(image_files)}] {title} ({year})")
        try:
            palette = extract_palette(img_path, n_clusters=6)
            profiles.append({
                "id":        content_id,
                "title":     title,
                "year":      year,
                "filename":  fname,
                "image_url": image_url,
                "palette":   palette,
            })
            print(f"  🎨 Dominant colour: RGB {palette[0]['rgb']}")
        except Exception as e:
            print(f"  ✗ Skipped ({e})")

    return profiles


# ── Mode 3: enrich existing profiles with descriptions + metadata ─────────────
def enrich(profiles: list[dict]) -> list[dict]:
    """
    For each profile in profiles.json, fetch the WikiArt detail endpoint
    and add: description, style, genre, tags, galleryName, sizeX, sizeY.
    Already-enriched profiles are skipped unless their description is null.
    """
    detail_url = "https://www.wikiart.org/en/App/Painting/ImageJson/{content_id}"

    enriched = []
    for i, profile in enumerate(profiles, 1):
        title = profile.get("title", "Untitled")
        content_id = profile.get("id", "")
        print(f"[{i}/{len(profiles)}] {title}")

        # Skip if already enriched and has a description
        if profile.get("description") is not None:
            print(f"  ✓ Already enriched")
            enriched.append(profile)
            continue

        # Only numeric IDs map to the detail endpoint
        if not str(content_id).isdigit():
            print(f"  – No numeric ID, skipping")
            enriched.append(profile)
            continue

        try:
            resp = requests.get(
                detail_url.format(content_id=content_id),
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            detail = resp.json()

            profile["description"] = detail.get("description")  # may be None
            profile["style"]       = detail.get("style")
            profile["genre"]       = detail.get("genre")
            profile["tags"]        = detail.get("tags")
            profile["gallery"]     = detail.get("galleryName")
            profile["size_cm"]     = (
                f"{detail['sizeX']} × {detail['sizeY']} cm"
                if detail.get("sizeX") and detail.get("sizeY") else None
            )

            if profile["description"]:
                preview = profile["description"][:80].rstrip() + "…"
                print(f"  📝 {preview}")
            else:
                print(f"  – No description available")

            time.sleep(DOWNLOAD_DELAY)

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        enriched.append(profile)

    return enriched


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reindex_mode = "--reindex" in sys.argv
    enrich_mode  = "--enrich"  in sys.argv

    paintings = fetch_painting_list()

    if enrich_mode:
        print("\n── Enrich mode: fetching descriptions + metadata ──")
        if not PROFILES_PATH.exists():
            print("✗ profiles.json not found. Run without flags first.")
            sys.exit(1)
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        profiles = enrich(profiles)
    elif reindex_mode:
        print("\n── Reindex mode: rebuilding profiles from existing images ──")
        profiles = reindex(paintings)
    else:
        print(f"\n── Download mode: fetching all {len(paintings)} paintings ──")
        profiles = download_all(paintings)

    if not profiles:
        print("\n✗ No paintings profiled. Nothing to save.")
        sys.exit(1)

    PROFILES_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅ {len(profiles)} paintings saved to {PROFILES_PATH}")

    if enrich_mode:
        print("\nDone! Restart the app to see descriptions.")
        print("  streamlit run app.py")
    elif not reindex_mode:
        print("\nNow open data/images/ in Finder, delete what you don't want,")
        print("then run:  python setup_dataset.py --reindex")
    else:
        print("\nDone! Restart the app to pick up the new dataset.")
        print("  streamlit run app.py")


if __name__ == "__main__":
    main()
