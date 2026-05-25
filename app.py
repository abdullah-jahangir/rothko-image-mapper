"""
app.py — rothpic
Upload any image → find the closest Rothko painting by color.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from utils.color_extractor import extract_palette
from utils.matcher import find_matches, load_profiles

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rothko Image Mapper",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
/* ── Dark background ── */
    [data-testid="stAppViewContainer"] {
        background-color: #111111;
        color: #e0e0e0;
    }
    [data-testid="stHeader"] { background-color: #111111; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; }

    /* ── File uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 4px;
    }
    [data-testid="stFileUploaderDropzone"]::before {
        content: "upload";
        display: block;
        text-align: center;
        color: #888;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        padding: 0.5rem 0 0.25rem 0;
    }
    /* Hide ALL instruction text inside the dropzone (drag/drop, limit, upload labels) */
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div > div {
        display: none !important;
    }
    /* Keep the Browse button visible */
    [data-testid="stFileUploaderDropzone"] button {
        display: inline-flex !important;
    }
    /* Hide the filename badge after upload */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"] {
        display: none !important;
    }

    /* ── General text ── */
    p, h1, h2, h3, label, div { color: #e0e0e0 !important; }

    /* ── Remove top padding ── */
    .block-container { padding-top: 2rem; }

    /* ── Arrow buttons — no box, just text ── */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #666 !important;
        font-size: 1rem !important;
        padding: 0.2rem 0.5rem !important;
        font-family: inherit !important;
    }
    div[data-testid="stButton"] button:hover:not(:disabled),
    div[data-testid="stDownloadButton"] button:hover {
        color: #ccc !important;
        background: transparent !important;
    }
    div[data-testid="stButton"] button:disabled {
        color: #333 !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
PROFILES_PATH = DATA_DIR / "profiles.json"
IMG_DIR = DATA_DIR / "images"


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data
def _load_profiles() -> list[dict]:
    return load_profiles(PROFILES_PATH)


@st.cache_data
def _extract_palette_cached(file_hash: str, file_bytes: bytes) -> list[dict]:
    """Cache keyed by MD5 hash so re-uploading the same file is instant."""
    return extract_palette(file_bytes)


# ── Text helpers ─────────────────────────────────────────────────────────────
import re

def clean_wikiart_markup(text: str) -> str:
    """Convert WikiArt [i]...[/i] tags to proper italics and strip other markup."""
    text = re.sub(r'\[i\](.*?)\[/i\]', r'<em>\1</em>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)  # strip any remaining unknown tags
    return text.strip()


# ── Centered image helper ────────────────────────────────────────────────────
import base64
import io as _io

def st_image_centered(img: Image.Image, display_w: int, href: str | None = None):
    """Render a PIL image centered in its column at a fixed pixel width.
    If href is given, the image is wrapped in a hyperlink."""
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_tag = (
        f'<img src="data:image/jpeg;base64,{b64}" width="{display_w}" '
        f'style="border-radius:2px;">'
    )
    inner = (
        f'<a href="{href}" target="_blank" '
        f'style="display:inline-block; cursor:pointer;">{img_tag}</a>'
        if href else img_tag
    )
    st.markdown(
        f'<div style="display:flex; justify-content:center;">{inner}</div>',
        unsafe_allow_html=True,
    )


def wikiart_page_url(image_url: str) -> str:
    """Derive the WikiArt painting page URL from its CDN image URL.
    e.g. https://uploads6.wikiart.org/images/mark-rothko/no-10.jpg!Large.jpg
         → https://www.wikiart.org/en/mark-rothko/no-10
    """
    slug = image_url.split("/")[-1].split("!")[0].rsplit(".", 1)[0]
    return f"https://www.wikiart.org/en/mark-rothko/{slug}"


# ── Swatch rendering ─────────────────────────────────────────────────────────
def make_swatch_strip(palette: list[dict], swatch_w: int = 60, swatch_h: int = 40) -> np.ndarray:
    """Return a horizontal numpy image strip of colour swatches."""
    swatches = []
    for color in palette:
        r, g, b = color["rgb"]
        tile = np.zeros((swatch_h, swatch_w, 3), dtype=np.uint8)
        tile[:, :] = [r, g, b]
        swatches.append(tile)
    return np.hstack(swatches)


# ── Match card ───────────────────────────────────────────────────────────────
def render_match(match: dict, user_palette: list[dict], label: str, is_primary: bool = False):
    """Render one match card: image + swatches + caption."""
    img_path = IMG_DIR / match["filename"]

    col_user, col_rothko = st.columns(2, gap="large")

    with col_user:
        if is_primary:
            st.markdown("**Your image**")
        # Swatch strip for uploaded image
        strip = make_swatch_strip(user_palette)
        st.image(strip, use_container_width=True)

    with col_rothko:
        similarity = match["similarity"]
        year = match.get("year", "")
        title = match.get("title", "Untitled")
        caption = f"**{title}** ({year}) — {similarity:.0f}% match"
        st.markdown(caption)

        if img_path.exists():
            rothko_img = Image.open(img_path).convert("RGB")
            st.image(rothko_img, width='stretch')
        else:
            st.warning(f"Image not found locally: {match['filename']}")

        # Swatch strip for Rothko painting
        strip = make_swatch_strip(match["palette"])
        st.image(strip, width='stretch')


def make_pairing_image(
    user_img: Image.Image,
    rothko_img: Image.Image,
    user_palette: list[dict],
    match_palette: list[dict],
    title: str,
    year: str | int,
) -> bytes:
    """Composite a side-by-side PNG of the user image and matched Rothko."""
    target_h = 500
    pad = 40          # padding around and between images
    swatch_h = 30     # height of colour strip
    bg = (17, 17, 17) # #111111

    left  = _match_height(user_img, target_h)
    right = _match_height(rothko_img, target_h)

    total_w = pad + left.width + pad + right.width + pad
    total_h = pad + target_h + 8 + swatch_h + pad

    canvas = Image.new("RGB", (total_w, total_h), bg)

    # Paste images
    canvas.paste(left,  (pad, pad))
    canvas.paste(right, (pad + left.width + pad, pad))

    # Swatch strips
    def paste_swatches(palette, x_start, img_w):
        sw = img_w // len(palette)
        for i, c in enumerate(palette):
            swatch = Image.new("RGB", (sw, swatch_h), tuple(c["rgb"]))
            canvas.paste(swatch, (x_start + i * sw, pad + target_h + 8))

    paste_swatches(user_palette,  pad, left.width)
    paste_swatches(match_palette, pad + left.width + pad, right.width)

    buf = _io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _match_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize an image to target_h pixels tall, preserving aspect ratio."""
    w, h = img.size
    new_w = max(1, int(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def render_side_by_side(user_img: Image.Image, user_palette: list[dict], match: dict):
    """Render the primary match as a full side-by-side comparison."""
    img_path = IMG_DIR / match["filename"]
    similarity = match["similarity"]
    year = match.get("year", "")
    title = match.get("title", "Untitled")

    # Cap display height at 500px so tall uploads don't dominate the page
    display_h = min(user_img.height, 500)
    user_display = _match_height(user_img, display_h)

    col_user, col_rothko = st.columns(2, gap="large")

    with col_user:
        st.markdown("#### Your image")
        st_image_centered(user_display, user_display.width)
        strip_img = Image.fromarray(make_swatch_strip(user_palette))
        st_image_centered(strip_img, user_display.width)
        st.caption("Dominant colors extracted")

    with col_rothko:
        st.markdown(f"#### {title} ({year})")
        if img_path.exists():
            rothko_img = Image.open(img_path).convert("RGB")
            rothko_display = _match_height(rothko_img, display_h)
            st_image_centered(rothko_display, rothko_display.width)
        else:
            st.warning(f"Image not found: {match['filename']}")
            rothko_display = None
        swatch_w = rothko_display.width if rothko_display else user_display.width
        strip_img = Image.fromarray(make_swatch_strip(match["palette"]))
        st_image_centered(strip_img, swatch_w)
        st.caption(f"{similarity:.0f}% color similarity")

        # ── Metadata ─────────────────────────────────────────────────────────
        meta_parts = []
        if match.get("style"):
            meta_parts.append(match["style"])
        if match.get("gallery"):
            meta_parts.append(f"📍 {match['gallery']}")
        if match.get("size_cm"):
            meta_parts.append(match["size_cm"])
        if meta_parts:
            st.caption("  ·  ".join(meta_parts))

        if match.get("tags"):
            tag_list = [t.strip() for t in match["tags"].split(",") if t.strip()]
            st.markdown(
                " ".join(
                    f"<span style='background:#2a2a2a; color:#aaa; "
                    f"padding:2px 8px; border-radius:12px; font-size:0.75rem;'>{t}</span>"
                    for t in tag_list
                ),
                unsafe_allow_html=True,
            )

        if match.get("description"):
            st.markdown("<br>", unsafe_allow_html=True)
            clean = clean_wikiart_markup(match["description"])
            st.markdown(
                f"<p style='color:#bbb; font-size:0.9rem; line-height:1.6;'>"
                f"{clean}</p>",
                unsafe_allow_html=True,
            )


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    # Title
    st.markdown(
        "<h1 style='text-align:center; letter-spacing:0.05em; "
        "font-weight:700; font-size:2.5rem;'>Rothko Image Mapper</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Check dataset exists
    if not PROFILES_PATH.exists():
        st.error(
            "Dataset not found. Run `python setup_dataset.py` first to download "
            "the Rothko paintings, then restart the app."
        )
        st.stop()

    profiles = _load_profiles()
    if not profiles:
        st.error("profiles.json is empty. Re-run `python setup_dataset.py`.")
        st.stop()

    # File uploader — centered
    _, upload_col, _ = st.columns([1, 2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    if uploaded is None:
        return

    # Process upload
    file_bytes = uploaded.getvalue()
    fhash = hashlib.md5(file_bytes).hexdigest()

    with st.spinner("Extracting colors…"):
        user_palette = _extract_palette_cached(fhash, file_bytes)

    with st.spinner("Finding your Rothko…"):
        matches = find_matches(user_palette, profiles, top_n=3)

    if not matches:
        st.error("No matches found. The dataset may be empty.")
        return

    user_img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")

    # ── Reset carousel when a new image is uploaded ──────────────────────────
    if st.session_state.get("carousel_file") != fhash:
        st.session_state["carousel_file"] = fhash
        st.session_state["carousel_idx"] = 0

    idx = st.session_state["carousel_idx"]
    match = matches[idx]

    st.markdown("---")

    display_h = min(user_img.height, 500)
    user_display = _match_height(user_img, display_h)

    # ── 4-column layout: user | ← | rothko | → ──────────────────────────────
    col_user, col_prev, col_rothko, col_next = st.columns([10, 1, 10, 1])

    # Left: uploaded image (fixed)
    with col_user:
        st.markdown("#### Your image")
        st_image_centered(user_display, user_display.width)
        strip_img = Image.fromarray(make_swatch_strip(user_palette))
        st_image_centered(strip_img, user_display.width)
        st.caption("Dominant colors extracted")

    # Right: current Rothko match
    img_path = IMG_DIR / match["filename"]
    title = match.get("title", "Untitled")
    year  = match.get("year", "")
    sim   = match["similarity"]
    rank  = ["Closest", "2nd closest", "3rd closest"][idx]

    if img_path.exists():
        rothko_img     = Image.open(img_path).convert("RGB")
        rothko_display = _match_height(rothko_img, display_h)
        swatch_w       = rothko_display.width
    else:
        rothko_display = None
        swatch_w       = user_display.width

    # Push arrows down to roughly the vertical midpoint of the image
    arrow_spacer = "<br>" * max(1, display_h // 55)

    with col_prev:
        st.markdown(arrow_spacer, unsafe_allow_html=True)
        if st.button("←", disabled=(idx == 0), use_container_width=True):
            st.session_state["carousel_idx"] -= 1
            st.rerun()

    with col_rothko:
        st.markdown(f"#### {title} ({year})")
        if rothko_display:
            page_url = wikiart_page_url(match.get("image_url", ""))
            st_image_centered(rothko_display, swatch_w, href=page_url)
        else:
            st.warning(f"Image not found: {match['filename']}")
        strip_img = Image.fromarray(make_swatch_strip(match["palette"]))
        st_image_centered(strip_img, swatch_w)

        # dots + caption on same line
        dots = "  ".join("⬤" if i == idx else "○" for i in range(len(matches)))
        st.markdown(
            f"<p style='color:#555; font-size:0.75rem; margin:0.2rem 0;'>"
            f"{dots}  ·  {rank}  ·  {sim:.0f}% match</p>",
            unsafe_allow_html=True,
        )

        # Download button
        if rothko_display:
            pairing_bytes = make_pairing_image(
                user_img, rothko_img,
                user_palette, match["palette"],
                title, year,
            )
            st.download_button(
                label="download pairing",
                data=pairing_bytes,
                file_name=f"rothko-match-{title.lower().replace(' ', '-')}.png",
                mime="image/png",
                use_container_width=True,
            )

    with col_next:
        st.markdown(arrow_spacer, unsafe_allow_html=True)
        if st.button("→", disabled=(idx == len(matches) - 1), use_container_width=True):
            st.session_state["carousel_idx"] += 1
            st.rerun()

    # Metadata (below col_rothko, outside the 4-col layout)
    with col_rothko:
        meta_parts = []
        if match.get("style"):
            meta_parts.append(match["style"])
        if match.get("gallery"):
            meta_parts.append(f"📍 {match['gallery']}")
        if match.get("size_cm"):
            meta_parts.append(match["size_cm"])
        if meta_parts:
            st.caption("  ·  ".join(meta_parts))

        if match.get("tags"):
            tag_list = [t.strip() for t in match["tags"].split(",") if t.strip()]
            st.markdown(
                " ".join(
                    f"<span style='background:#2a2a2a; color:#aaa; "
                    f"padding:2px 8px; border-radius:12px; font-size:0.75rem;'>{t}</span>"
                    for t in tag_list
                ),
                unsafe_allow_html=True,
            )

        if match.get("description"):
            st.markdown("<br>", unsafe_allow_html=True)
            clean = clean_wikiart_markup(match["description"])
            st.markdown(
                f"<p style='color:#bbb; font-size:0.9rem; line-height:1.6;'>{clean}</p>",
                unsafe_allow_html=True,
            )


def render_footer():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div style='max-width:680px; margin:0 auto; padding:1.5rem 0 2.5rem 0;
                    color:#555; font-size:1rem; line-height:1.9; text-align:center;'>
            <p style='color:#444; letter-spacing:0.08em; font-size:0.85rem;
                      margin-bottom:1rem;'>HOW IT WORKS</p>
            <p>
                Your photo is reduced to six dominant colours using
                <em>K-means clustering</em>, which partitions all pixels in the image
                into groups by proximity in colour space, then takes the centroid of each group.
            </p>
            <p>
                Each centroid is expressed in <em>CIELAB colour space</em>, a perceptually
                uniform model where equal numerical distances correspond to equal perceived
                colour differences. Those six values are compared against pre-computed
                palettes for 100 Rothko paintings using weighted nearest-neighbour distance.
                The closest painting is your match.
            </p>
            <p>
                Paintings sourced from <a href="https://www.wikiart.org/en/mark-rothko"
                target="_blank" style="color:#666;">WikiArt</a>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    render_footer()
