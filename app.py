# ==========================================
# IMPORTS
# ==========================================
from __future__ import annotations

import base64
import html
import io
import math
import os
import re
import uuid
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Boolean, Column, Date, DateTime, Integer, String, Text, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

try:  # optional: lets you override settings (e.g. HERO_IMAGE_URL) from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# ==========================================
# CONFIGURATION
# ==========================================
APP_NAME = "Camera Clicks Photography"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "photography.db"
PHOTOS_DIR = BASE_DIR / "photos"
EDITED_DIR = BASE_DIR / "edited_photos"

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}
MAX_UPLOAD_MB = 25
MAX_STORED_SIDE = 3000
EDITOR_WORK_SIDE = 2400

HERO_IMAGE_URL = os.getenv(
    "HERO_IMAGE_URL",
    "https://images.unsplash.com/photo-1452587925148-ce544e77e70d?auto=format&fit=crop&w=1920&q=80",
)

CATEGORIES = [
    "Nature", "Portrait", "Landscape", "Wildlife", "Travel", "Street",
    "Architecture", "Events", "Food", "Night Photography", "Black & White", "Other",
]
CATEGORY_EMOJI = {
    "Nature": "🌿", "Portrait": "🧑‍🎨", "Landscape": "🏔️", "Wildlife": "🦁", "Travel": "✈️",
    "Street": "🏙️", "Architecture": "🏛️", "Events": "🎉", "Food": "🍽️",
    "Night Photography": "🌃", "Black & White": "🎞️", "Other": "✨",
}
PAGES = ["🏠 Home", "📸 Capture", "🖼️ Gallery", "❤️ Favorites", "✨ Photo Editor", "📊 Analytics", "ℹ️ About"]
PAGE_HOME, PAGE_CAPTURE, PAGE_GALLERY, PAGE_FAVORITES, PAGE_EDITOR, PAGE_ANALYTICS, PAGE_ABOUT = PAGES
SORT_OPTIONS = ["Newest first", "Oldest first", "Title A-Z"]

EDITOR_DEFAULTS = {
    "ed_brightness": 1.0, "ed_contrast": 1.0, "ed_sharpness": 1.0, "ed_saturation": 1.0,
    "ed_grayscale": False, "ed_blur": 0, "ed_rotation": 0, "ed_resize": 100,
}

st.set_page_config(page_title=APP_NAME, page_icon="📷", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# DATABASE SETUP
# ==========================================
Base = declarative_base()


@st.cache_resource(show_spinner=False)
def get_session_factory():
    """Create the SQLite engine once, create tables if missing, return a session factory."""
    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


# ==========================================
# SQLALCHEMY MODELS
# ==========================================
class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(150), nullable=False)
    description = Column(Text, default="")
    category = Column(String(50), default="Other")
    location = Column(String(150), default="")
    tags = Column(String(300), default="")
    photo_date = Column(Date, default=date.today)
    filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    is_favorite = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


# ==========================================
# DATABASE FUNCTIONS
# ==========================================
def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def clean_text(text: Any, limit: int) -> str:
    return str(text or "").strip()[:limit]


def clean_tags(text: Any) -> str:
    seen, result = set(), []
    for raw in re.split(r"[,;]", str(text or "")):
        tag = raw.strip().lstrip("#").strip().lower()[:30]
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return ", ".join(result[:12])


def photo_to_dict(p: Photo) -> dict:
    tags = p.tags or ""
    return {
        "id": p.id,
        "title": p.title or "Untitled",
        "description": p.description or "",
        "category": p.category or "Other",
        "location": p.location or "",
        "tags": tags,
        "tag_list": [t.strip() for t in tags.split(",") if t.strip()],
        "photo_date": p.photo_date,
        "filename": p.filename,
        "image_path": p.image_path,
        "is_favorite": bool(p.is_favorite),
        "created_at": p.created_at,
    }


def add_photo(title, description, category, location, tags, photo_date, filename, image_path) -> Optional[int]:
    try:
        with get_session_factory()() as session:
            photo = Photo(
                title=clean_text(title, 120) or "Untitled",
                description=clean_text(description, 1000),
                category=category if category in CATEGORIES else "Other",
                location=clean_text(location, 150),
                tags=clean_tags(tags),
                photo_date=photo_date or date.today(),
                filename=filename,
                image_path=image_path,
                is_favorite=False,
                created_at=datetime.now(),
            )
            session.add(photo)
            session.commit()
            return photo.id
    except SQLAlchemyError as exc:
        st.error(f"Database error while saving the photograph ({exc.__class__.__name__}).")
        return None


def get_all_photos() -> list[dict]:
    try:
        with get_session_factory()() as session:
            rows = session.scalars(select(Photo).order_by(Photo.created_at.desc(), Photo.id.desc())).all()
            return [photo_to_dict(r) for r in rows]
    except SQLAlchemyError as exc:
        st.error(f"Database error while reading photographs ({exc.__class__.__name__}).")
        return []


def get_photo(photo_id: int) -> Optional[dict]:
    try:
        with get_session_factory()() as session:
            row = session.get(Photo, photo_id)
            return photo_to_dict(row) if row else None
    except SQLAlchemyError:
        return None


def update_photo(photo_id: int, **fields) -> bool:
    allowed = {"title", "description", "category", "location", "tags", "photo_date"}
    try:
        with get_session_factory()() as session:
            row = session.get(Photo, photo_id)
            if row is None:
                st.error("That photograph no longer exists.")
                return False
            for key, value in fields.items():
                if key in allowed:
                    setattr(row, key, value)
            session.commit()
            return True
    except SQLAlchemyError as exc:
        st.error(f"Database error while updating ({exc.__class__.__name__}).")
        return False


def delete_photo(photo_id: int) -> bool:
    try:
        with get_session_factory()() as session:
            row = session.get(Photo, photo_id)
            if row is None:
                return False
            rel_path = row.image_path
            session.delete(row)
            session.commit()
            still_used = session.scalar(select(func.count()).select_from(Photo).where(Photo.image_path == rel_path))
        if not still_used:
            with suppress(OSError):
                abs_path(rel_path).unlink()
        return True
    except SQLAlchemyError as exc:
        st.error(f"Database error while deleting ({exc.__class__.__name__}).")
        return False


def toggle_favorite(photo_id: int) -> Optional[bool]:
    try:
        with get_session_factory()() as session:
            row = session.get(Photo, photo_id)
            if row is None:
                return None
            row.is_favorite = not bool(row.is_favorite)
            session.commit()
            return bool(row.is_favorite)
    except SQLAlchemyError as exc:
        st.error(f"Database error while updating favorites ({exc.__class__.__name__}).")
        return None


# ==========================================
# IMAGE STORAGE FUNCTIONS
# ==========================================
class ImageError(Exception):
    """Friendly, user-facing image problem."""


def ensure_directories() -> None:
    for folder in (PHOTOS_DIR, EDITED_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def abs_path(relative: str) -> Path:
    return BASE_DIR / relative


def slugify(text: str, default: str = "photo") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:40]
    return slug or default


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def validate_upload(name: str, data: bytes) -> None:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ImageError(f"Unsupported file type '.{ext}'. Please use JPG, JPEG, PNG or WEBP.")
    if not data:
        raise ImageError("The file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageError(f"The file is larger than {MAX_UPLOAD_MB} MB.")


def decode_image(data: bytes) -> Image.Image:
    """Validate and decode image bytes into an RGB PIL image (raises ImageError)."""
    if not data:
        raise ImageError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise ImageError(f"Unsupported image format ({fmt or 'unknown'}). Please use JPG, PNG or WEBP.")
        img = Image.open(io.BytesIO(data))
        img.load()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ImageError("This file is not a valid image or it is corrupted.") from None
    return to_rgb(ImageOps.exif_transpose(img))


def save_pil_image(img: Image.Image, name_hint: str, directory: Path = PHOTOS_DIR) -> str:
    """Save a PIL image as a safely named JPEG. Never overwrites: every name is unique."""
    try:
        ensure_directories()
    except OSError as exc:
        raise ImageError(f"Could not create the storage folders: {exc}") from exc
    img = to_rgb(img)
    if max(img.size) > MAX_STORED_SIDE:
        img = img.copy()
        img.thumbnail((MAX_STORED_SIDE, MAX_STORED_SIDE), Image.Resampling.LANCZOS)
    filename = f"{slugify(name_hint)}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.jpg"
    try:
        img.save(directory / filename, format="JPEG", quality=92, optimize=True)
    except OSError as exc:
        raise ImageError(f"Could not save the image: {exc}") from exc
    return filename


def load_pil(path: Path, max_side: Optional[int] = None) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            img = to_rgb(ImageOps.exif_transpose(im))
        if max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return img
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None


def pil_to_data_uri(img: Image.Image, max_side: int = 1200, quality: int = 85) -> str:
    img = to_rgb(img).copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False, max_entries=512)
def image_data_uri(path_str: str, mtime: float, max_side: int) -> str:
    img = load_pil(Path(path_str), max_side)
    return pil_to_data_uri(img, max_side) if img is not None else ""


def photo_uri(photo: dict, max_side: int = 700) -> str:
    path = abs_path(photo["image_path"])
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    return image_data_uri(str(path), mtime, max_side)


def read_file_bytes(photo: dict) -> Optional[bytes]:
    try:
        return abs_path(photo["image_path"]).read_bytes()
    except OSError:
        return None


# ==========================================
# IMAGE EDITING FUNCTIONS
# ==========================================
def apply_edits(img, brightness=1.0, contrast=1.0, sharpness=1.0, saturation=1.0,
                grayscale=False, blur=0, rotation=0, resize_pct=100) -> Image.Image:
    """Return an edited COPY of the image. The original is never modified."""
    out = to_rgb(img).copy()
    if rotation % 360:
        out = out.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    if resize_pct != 100:
        w, h = out.size
        scale = resize_pct / 100.0
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        out = out.resize(new_size, Image.Resampling.LANCZOS)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(saturation)
    out = ImageEnhance.Sharpness(out).enhance(sharpness)
    if blur and blur > 0:
        kernel = 2 * int(blur) + 1
        out = Image.fromarray(cv2.GaussianBlur(np.array(out), (kernel, kernel), 0))
    if grayscale:
        gray = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2GRAY)
        out = Image.fromarray(gray).convert("RGB")
    return out


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    if fmt.upper() == "PNG":
        to_rgb(img).save(buffer, format="PNG")
    else:
        to_rgb(img).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=1024)
def image_features(path_str: str, mtime: float) -> Optional[np.ndarray]:
    """Colour-histogram feature vector used for the similar-photo feature."""
    try:
        img = load_pil(Path(path_str), 128)
        if img is None:
            return None
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        norm = np.linalg.norm(hist)
        return (hist / norm if norm > 0 else hist).astype(np.float32)
    except Exception:
        return None


def _features_for(photo: dict) -> Optional[np.ndarray]:
    path = abs_path(photo["image_path"])
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return image_features(str(path), mtime)


# ==========================================
# SEARCH / FILTER FUNCTIONS
# ==========================================
def filter_photos(photos, query="", categories=None, location="All locations", tags=None,
                  date_range=None, favorites_only=False, sort_by="Newest first") -> list[dict]:
    result = list(photos)
    tokens = [t for t in query.lower().split() if t]
    if tokens:
        def haystack(p):
            return " ".join([p["title"], p["description"], p["category"], p["location"], p["tags"]]).lower()
        result = [p for p in result if all(t in haystack(p) for t in tokens)]
    if categories:
        result = [p for p in result if p["category"] in categories]
    if location and location != "All locations":
        result = [p for p in result if p["location"] == location]
    if tags:
        result = [p for p in result if any(t in p["tag_list"] for t in tags)]
    if date_range:
        start, end = date_range
        result = [p for p in result if p["photo_date"] and start <= p["photo_date"] <= end]
    if favorites_only:
        result = [p for p in result if p["is_favorite"]]
    if sort_by == "Oldest first":
        result.sort(key=lambda p: (p["photo_date"] or date.min, p["id"]))
    elif sort_by == "Title A-Z":
        result.sort(key=lambda p: p["title"].lower())
    else:
        result.sort(key=lambda p: (p["photo_date"] or date.min, p["id"]), reverse=True)
    return result


def similar_photos(target: dict, photos: list[dict], k: int = 3) -> list[dict]:
    target_vec = _features_for(target)
    if target_vec is None:
        return []
    candidates, vectors = [], []
    for p in photos:
        if p["id"] == target["id"]:
            continue
        vec = _features_for(p)
        if vec is not None:
            candidates.append(p)
            vectors.append(vec)
    if not candidates:
        return []
    sims = cosine_similarity(target_vec.reshape(1, -1), np.vstack(vectors))[0]
    return [candidates[i] for i in np.argsort(-sims)[:k]]


def recommend_from_favorites(photos: list[dict], favorites: list[dict], k: int = 3) -> list[dict]:
    fav_vecs = [v for v in (_features_for(p) for p in favorites) if v is not None]
    if not fav_vecs:
        return []
    candidates, vectors = [], []
    for p in photos:
        if p["is_favorite"]:
            continue
        vec = _features_for(p)
        if vec is not None:
            candidates.append(p)
            vectors.append(vec)
    if not candidates:
        return []
    centroid = np.mean(np.vstack(fav_vecs), axis=0).reshape(1, -1)
    sims = cosine_similarity(centroid, np.vstack(vectors))[0]
    return [candidates[i] for i in np.argsort(-sims)[:k]]


# ==========================================
# ANALYTICS FUNCTIONS
# ==========================================
PANEL, ACCENT, MUTED, TEXT_COLOR = "#141417", "#e3b45f", "#9a9aa3", "#f1f0ec"


def photos_to_dataframe(photos: list[dict]) -> pd.DataFrame:
    if not photos:
        return pd.DataFrame()
    df = pd.DataFrame(photos)
    df["photo_date"] = pd.to_datetime(df["photo_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["location"] = df["location"].replace("", "Unknown").fillna("Unknown")
    return df


def _new_fig(width=6.0, height=3.6):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#2a2a31")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(TEXT_COLOR)
    return fig, ax


def chart_by_category(df: pd.DataFrame):
    counts = df["category"].value_counts().sort_values()
    fig, ax = _new_fig()
    bars = ax.barh(counts.index, counts.values, color=ACCENT, height=0.6)
    ax.bar_label(bars, color=TEXT_COLOR, padding=4, fontsize=9)
    ax.set_title("Photos by Category", fontsize=12, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, max(counts.values) * 1.2)
    fig.tight_layout()
    return fig


def chart_by_month(df: pd.DataFrame):
    months = df["created_at"].dropna().dt.to_period("M")
    if months.empty:
        return None
    counts = months.value_counts().sort_index()
    labels = [str(m) for m in counts.index]
    x = np.arange(len(labels))
    fig, ax = _new_fig()
    ax.bar(x, counts.values, color=ACCENT, width=0.55, label="Uploaded")
    ax.plot(x, np.cumsum(counts.values), color="#ffffff", marker="o", linewidth=1.6, label="Cumulative")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Photos Uploaded by Month", fontsize=12, pad=12)
    legend = ax.legend(facecolor=PANEL, edgecolor="#2a2a31", fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    fig.tight_layout()
    return fig


def chart_by_location(df: pd.DataFrame):
    counts = df["location"].value_counts().head(10).sort_values()
    fig, ax = _new_fig()
    bars = ax.barh(counts.index, counts.values, color="#c98a2b", height=0.6)
    ax.bar_label(bars, color=TEXT_COLOR, padding=4, fontsize=9)
    ax.set_title("Photos by Location (Top 10)", fontsize=12, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, max(counts.values) * 1.2)
    fig.tight_layout()
    return fig


def chart_favorites(df: pd.DataFrame):
    fav = int(df["is_favorite"].sum())
    other = int(len(df) - fav)
    values = [(fav, "Favorites", ACCENT), (other, "Not favorite", "#3a3a42")]
    values = [v for v in values if v[0] > 0]
    fig, ax = _new_fig()
    ax.pie(
        [v[0] for v in values], labels=[v[1] for v in values], colors=[v[2] for v in values],
        autopct="%1.0f%%", startangle=90, wedgeprops={"width": 0.42, "edgecolor": PANEL},
        textprops={"color": TEXT_COLOR, "fontsize": 10},
    )
    ax.set_title("Favorite vs Non-Favorite", fontsize=12, pad=12)
    ax.axis("equal")
    fig.tight_layout()
    return fig


# ==========================================
# CUSTOM CSS
# ==========================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#0a0a0c;--panel:#141417;--panel2:#1a1a1f;--line:#2a2a31;--text:#f1f0ec;--muted:#9a9aa3;--accent:#e3b45f;--accent2:#c98a2b;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--bg) !important;color:var(--text);font-family:'Inter',system-ui,sans-serif;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:4rem;}
h1,h2,h3,h4{font-family:'Playfair Display',Georgia,serif !important;color:var(--text) !important;letter-spacing:.3px;}
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], .stCheckbox label, .stToggle label, .stRadio label{color:var(--text);}
[data-testid="stCaptionContainer"]{color:var(--muted);}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e0e11,#070708);border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--text);}
.brand{padding:10px 6px 18px 6px;border-bottom:1px solid var(--line);margin-bottom:14px;}
.brand .logo{font-size:2rem;}
.brand .name{font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:800;letter-spacing:2px;color:var(--accent);line-height:1.1;}
.brand .sub{font-size:.7rem;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-top:4px;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px;}
[data-testid="stSidebar"] div[role="radiogroup"] label{border:1px solid transparent;border-radius:12px;padding:10px 14px;transition:all .2s;width:100%;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#17171b;border-color:var(--line);}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(227,180,95,.20),rgba(227,180,95,.04));border-color:rgba(227,180,95,.55);}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{display:none;}
.side-stats{margin-top:18px;padding:14px;border:1px solid var(--line);border-radius:14px;background:var(--panel);font-size:.85rem;color:var(--muted);}
.side-stats b{color:var(--accent);font-size:1.1rem;}
/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"]{
 width:100%;border-radius:999px;border:1px solid #3a3a42;background:var(--panel2);color:var(--text);font-weight:500;padding:.5rem 1rem;transition:all .2s;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px);box-shadow:0 6px 18px rgba(0,0,0,.45);}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"], .stDownloadButton > button[kind="primary"]{
 background:linear-gradient(135deg,var(--accent),var(--accent2));color:#16110a !important;border:none;font-weight:600;}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{color:#000 !important;filter:brightness(1.08);}
/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input,
.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div{
 background:#17171b !important;color:var(--text) !important;border:1px solid var(--line) !important;border-radius:12px !important;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] div[role="listbox"]{background:#17171b !important;color:var(--text) !important;}
div[data-baseweb="popover"] li{color:var(--text) !important;}
[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"]{background:var(--panel);border:1px dashed #3a3a42;border-radius:16px;}
[data-testid="stFileUploaderDropzone"] *{color:var(--text);}
[data-testid="stForm"], [data-testid="stExpander"]{background:var(--panel);border:1px solid var(--line) !important;border-radius:16px;}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{color:var(--muted);border-radius:10px 10px 0 0;padding:10px 18px;}
.stTabs [aria-selected="true"]{color:var(--accent) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent) !important;}
/* Hero */
.hero{position:relative;min-height:560px;border-radius:28px;overflow:hidden;display:flex;align-items:center;justify-content:center;text-align:center;border:1px solid var(--line);
 background-image:linear-gradient(180deg,rgba(8,8,10,.35),rgba(8,8,10,.9)),var(--hero-url, none),radial-gradient(circle at 20% 20%,#4a3416 0%,#16130f 50%,#070708 100%);
 background-size:cover;background-position:center;box-shadow:0 30px 80px rgba(0,0,0,.6);margin-bottom:18px;}
.hero-inner{padding:40px 24px;max-width:900px;}
.hero-kicker{letter-spacing:6px;font-size:.78rem;color:var(--accent);margin-bottom:18px;}
.hero-title{font-family:'Playfair Display',serif !important;font-size:5.4rem;font-weight:800;letter-spacing:10px;margin:0;color:#fff !important;text-shadow:0 6px 30px rgba(0,0,0,.7);line-height:1.05;}
.hero-sub{font-size:1.15rem;letter-spacing:8px;text-transform:uppercase;color:var(--accent);margin-top:10px;}
.hero-tag{font-size:1.15rem;color:#d9d7d0;margin:22px auto 0;max-width:640px;line-height:1.7;font-weight:300;}
/* Sections */
.sec-head{margin:46px 0 20px 0;}
.sec-kicker{font-size:.75rem;letter-spacing:5px;text-transform:uppercase;color:var(--accent);}
.sec-head h2{font-size:2.3rem;margin:6px 0 4px 0;}
.sec-head p{color:var(--muted);margin:0;}
.glass{background:linear-gradient(145deg,#17171b,#101013);border:1px solid var(--line);border-radius:20px;padding:26px;box-shadow:0 14px 40px rgba(0,0,0,.4);}
.glass p{color:#c9c8c2;line-height:1.8;}
.quote{font-family:'Playfair Display',serif;font-size:1.5rem;line-height:1.5;color:var(--accent);}
.cat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px;}
.cat-card{background:linear-gradient(145deg,#18181c,#101013);border:1px solid var(--line);border-radius:18px;padding:22px;transition:all .3s;}
.cat-card:hover{transform:translateY(-4px);border-color:var(--accent);box-shadow:0 14px 34px rgba(0,0,0,.5);}
.cat-card .emoji{font-size:2rem;}
.cat-card .cname{font-weight:600;margin-top:8px;}
.cat-card .ccount{color:var(--muted);font-size:.85rem;}
.feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;}
.cta{text-align:center;padding:54px 24px;border-radius:26px;border:1px solid rgba(227,180,95,.35);
 background:radial-gradient(circle at 50% 0%,rgba(227,180,95,.20),rgba(20,20,23,.95) 65%);margin:40px 0 18px 0;}
.cta h2{font-size:2.4rem;margin:0 0 8px 0;}
.cta p{color:var(--muted);}
/* Photo cards */
.photo-card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.45);transition:all .3s;margin-bottom:10px;}
.photo-card:hover{transform:translateY(-4px);border-color:rgba(227,180,95,.6);}
.photo-frame{position:relative;aspect-ratio:4/3;overflow:hidden;background:#0e0e10;}
.photo-frame img{width:100%;height:100%;object-fit:cover;transition:transform .7s ease;display:block;}
.photo-card:hover .photo-frame img{transform:scale(1.08);}
.fav-badge{position:absolute;top:10px;right:10px;background:rgba(10,10,12,.75);border:1px solid rgba(227,180,95,.5);color:var(--accent);padding:3px 10px;border-radius:999px;font-size:.72rem;backdrop-filter:blur(6px);}
.photo-body{padding:14px 16px 16px 16px;}
.photo-title{font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.photo-meta{margin-top:8px;font-size:.8rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 12px;}
.photo-missing{display:flex;align-items:center;justify-content:center;height:100%;min-height:160px;color:var(--muted);background:repeating-linear-gradient(45deg,#121215,#121215 10px,#17171b 10px,#17171b 20px);}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}
.chip{border:1px solid var(--line);background:#1a1a1f;border-radius:999px;padding:4px 12px;font-size:.78rem;color:#cfcfd4;}
.chip.gold{border-color:rgba(227,180,95,.6);color:var(--accent);background:rgba(227,180,95,.08);}
/* Detail */
.detail-frame{border-radius:22px;overflow:hidden;border:1px solid var(--line);box-shadow:0 24px 60px rgba(0,0,0,.6);background:#000;}
.detail-frame img{width:100%;display:block;}
.detail-title{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:800;line-height:1.15;margin-bottom:6px;}
.detail-meta{display:grid;grid-template-columns:1fr;gap:10px;margin:16px 0;}
.detail-meta div{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;}
.detail-meta span{display:block;font-size:.72rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}
.detail-desc{color:#c9c8c2;line-height:1.8;margin:10px 0 6px 0;}
/* Stats */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:18px;}
.stat-card{background:linear-gradient(145deg,#18181c,#101013);border:1px solid var(--line);border-radius:18px;padding:22px;}
.stat-card .num{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:800;color:var(--accent);}
.stat-card .lbl{color:var(--muted);font-size:.8rem;letter-spacing:2px;text-transform:uppercase;}
.empty{text-align:center;padding:60px 20px;border:1px dashed #3a3a42;border-radius:22px;background:var(--panel);margin:20px 0;}
.empty .big{font-size:3rem;}
.empty h3{margin:8px 0 4px 0;}
.empty p{color:var(--muted);}
.compare-label{font-size:.75rem;letter-spacing:4px;text-transform:uppercase;color:var(--accent);margin-bottom:8px;}
.footer-note{text-align:center;color:var(--muted);font-size:.8rem;margin-top:50px;letter-spacing:2px;}
@media (max-width:768px){
 .hero{min-height:420px;border-radius:20px;}
 .hero-title{font-size:2.5rem;letter-spacing:4px;}
 .hero-sub{font-size:.9rem;letter-spacing:5px;}
 .hero-tag{font-size:1rem;}
 .sec-head h2{font-size:1.7rem;}
 .detail-title{font-size:1.7rem;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section_heading(kicker: str, title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    html_block(f'<div class="sec-head"><div class="sec-kicker">{esc(kicker)}</div><h2>{esc(title)}</h2>{sub}</div>')


def fmt_date(value) -> str:
    return value.strftime("%d %b %Y") if value else "—"


def photo_card_html(p: dict, max_side: int = 700) -> str:
    uri = photo_uri(p, max_side)
    img = (f'<img src="{uri}" alt="{esc(p["title"])}" loading="lazy">' if uri
           else '<div class="photo-missing">Image unavailable</div>')
    badge = '<span class="fav-badge">❤️ Favorite</span>' if p["is_favorite"] else ""
    location = esc(p["location"]) or "Unknown location"
    return (
        '<div class="photo-card"><div class="photo-frame">' + img + badge + '</div><div class="photo-body">'
        f'<div class="photo-title">{esc(p["title"])}</div>'
        f'<div class="photo-meta"><span>🏷️ {esc(p["category"])}</span><span>📅 {fmt_date(p["photo_date"])}</span>'
        f'<span>📍 {location}</span></div></div></div>'
    )


def empty_state(icon: str, title: str, text: str, button_label: Optional[str] = None,
                target: Optional[str] = None, key: str = "empty") -> None:
    html_block(f'<div class="empty"><div class="big">{icon}</div><h3>{esc(title)}</h3><p>{esc(text)}</p></div>')
    if button_label and target:
        _, mid, _ = st.columns([1, 1, 1])
        mid.button(button_label, key=f"{key}_btn", type="primary", on_click=cb_go, args=(target,))


def flash(message: str, kind: str = "success") -> None:
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(message)


# ==========================================
# NAVIGATION
# ==========================================
def cb_go(page: str) -> None:
    st.session_state["nav_page"] = page


def cb_view_photo(photo_id: int) -> None:
    st.session_state["selected_photo_id"] = photo_id
    st.session_state["nav_page"] = PAGE_GALLERY


def cb_close_photo() -> None:
    st.session_state["selected_photo_id"] = None


def cb_toggle_favorite(photo_id: int) -> None:
    toggle_favorite(photo_id)


def cb_edit_photo(photo_id: int) -> None:
    st.session_state["ed_source"] = "Gallery photo"
    st.session_state["ed_pick"] = photo_id
    st.session_state["nav_page"] = PAGE_EDITOR


def cb_open_category() -> None:
    st.session_state["g_categories"] = [st.session_state.get("home_cat_pick", CATEGORIES[0])]
    st.session_state["selected_photo_id"] = None
    st.session_state["nav_page"] = PAGE_GALLERY


def cb_reset_filters() -> None:
    st.session_state.update({
        "g_query": "", "g_categories": [], "g_location": "All locations", "g_tags": [],
        "g_use_date": False, "g_fav_only": False, "g_sort": SORT_OPTIONS[0], "g_page": 1,
    })
    st.session_state.pop("g_dates", None)


def cb_reset_editor() -> None:
    for key, value in EDITOR_DEFAULTS.items():
        st.session_state[key] = value


def init_state() -> None:
    state = st.session_state
    state.setdefault("nav_page", PAGE_HOME)
    state.setdefault("selected_photo_id", None)
    state.setdefault("cam_n", 0)
    state.setdefault("up_n", 0)
    state.setdefault("g_query", "")
    state.setdefault("g_categories", [])
    state.setdefault("g_location", "All locations")
    state.setdefault("g_tags", [])
    state.setdefault("g_use_date", False)
    state.setdefault("g_fav_only", False)
    state.setdefault("g_sort", SORT_OPTIONS[0])
    state.setdefault("g_per_page", 9)
    state.setdefault("g_page", 1)
    state.setdefault("ed_source", "Gallery photo")
    for key, value in EDITOR_DEFAULTS.items():
        state.setdefault(key, value)


def render_sidebar() -> None:
    photos = get_all_photos()
    favs = sum(1 for p in photos if p["is_favorite"])
    with st.sidebar:
        html_block('<div class="brand"><div class="logo">📷</div><div class="name">CAMERA CLICKS</div>'
                   '<div class="sub">Photography</div></div>')
        st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        html_block(f'<div class="side-stats"><b>{len(photos)}</b> photographs<br><b>{favs}</b> favorites</div>')


def render_photo_grid(photos: list[dict], ctx: str, columns: int = 3) -> None:
    for start in range(0, len(photos), columns):
        cols = st.columns(columns)
        for col, p in zip(cols, photos[start:start + columns]):
            with col:
                html_block(photo_card_html(p))
                b1, b2 = st.columns(2)
                b1.button("🔍 View", key=f"{ctx}_view_{p['id']}", on_click=cb_view_photo, args=(p["id"],))
                b2.button("💔 Remove" if p["is_favorite"] else "🤍 Favorite", key=f"{ctx}_fav_{p['id']}",
                          on_click=cb_toggle_favorite, args=(p["id"],))


# ==========================================
# HOME PAGE
# ==========================================
def home_page() -> None:
    photos = get_all_photos()
    show_flash()
    hero_css = "url('" + HERO_IMAGE_URL + "')"
    html_block(
        f'<div class="hero" style="--hero-url:{hero_css}"><div class="hero-inner">'
        '<div class="hero-kicker">✦ PHOTOGRAPHY PORTFOLIO ✦</div>'
        '<h1 class="hero-title">CAMERA CLICKS</h1>'
        '<div class="hero-sub">Photography Website</div>'
        '<p class="hero-tag">Capturing moments. Creating memories. Telling stories through photography.</p>'
        '</div></div>'
    )
    b1, b2, b3 = st.columns(3)
    b1.button("🖼️ Explore Gallery", key="hero_gallery", type="primary", on_click=cb_go, args=(PAGE_GALLERY,))
    b2.button("📸 Capture Moments", key="hero_capture", on_click=cb_go, args=(PAGE_CAPTURE,))
    b3.button("❤️ View Portfolio", key="hero_portfolio", on_click=cb_go, args=(PAGE_FAVORITES,))

    section_heading("About Photography", "Light, Frame, Story",
                    "A photograph is a pause button for the moments that matter.")
    left, right = st.columns([3, 2])
    with left:
        html_block(
            '<div class="glass"><p>Camera Clicks is a home for your photographs. Whether you shoot sweeping landscapes, '
            'quiet portraits, busy streets or a perfect plate of food, every frame deserves to be seen, '
            'organised and remembered.</p><p>Capture with your camera, upload your best shots, polish them in the '
            'editor and build a gallery that feels like a real portfolio.</p></div>'
        )
    with right:
        html_block('<div class="glass"><div class="quote">“Photography is the story I fail to put into words.”</div>'
                   '<p style="margin-top:12px">— Destin Sparks</p></div>')

    section_heading("Featured Work", "Selected Photographs", "Your favorites first, then your latest captures.")
    featured = sorted(photos, key=lambda p: (not p["is_favorite"], -p["id"]))[:6]
    if featured:
        render_photo_grid(featured, "home", 3)
    else:
        empty_state("🖼️", "Your portfolio is waiting",
                    "Add your first photograph and it will be featured here.",
                    "📸 Add your first photo", PAGE_CAPTURE, "home_empty")

    section_heading("Photography Categories", "Explore by Genre", "Twelve ways to organise your work.")
    counts = pd.Series([p["category"] for p in photos], dtype="object").value_counts().to_dict() if photos else {}
    cards = "".join(
        f'<div class="cat-card"><div class="emoji">{CATEGORY_EMOJI[c]}</div><div class="cname">{esc(c)}</div>'
        f'<div class="ccount">{counts.get(c, 0)} photo{"s" if counts.get(c, 0) != 1 else ""}</div></div>'
        for c in CATEGORIES
    )
    html_block(f'<div class="cat-grid">{cards}</div>')
    c1, c2 = st.columns([3, 1])
    c1.selectbox("Browse a category", CATEGORIES, key="home_cat_pick", label_visibility="collapsed")
    c2.button("Open in Gallery", key="home_open_cat", on_click=cb_open_category)

    section_heading("Why Photography?", "Moments Don't Wait", "Three reasons to keep your camera close.")
    html_block(
        '<div class="feature-grid">'
        '<div class="glass"><h3>🕰️ Preserve Time</h3><p>Faces change, places evolve. A photograph keeps a moment exactly as it felt.</p></div>'
        '<div class="glass"><h3>💬 Tell Stories</h3><p>One image can say what paragraphs cannot, and it speaks every language.</p></div>'
        '<div class="glass"><h3>👁️ See Differently</h3><p>Photography trains you to notice light, colour and detail in everyday life.</p></div>'
        '</div>'
    )

    html_block('<div class="cta"><h2>Ready to capture your next story?</h2>'
               '<p>Browse the collection or add a new frame to your portfolio right now.</p></div>')
    _, m1, m2, _ = st.columns([1, 1.2, 1.2, 1])
    m1.button("🖼️ Explore Gallery", key="cta_gallery", type="primary", on_click=cb_go, args=(PAGE_GALLERY,))
    m2.button("📸 Capture Your Moment", key="cta_capture", on_click=cb_go, args=(PAGE_CAPTURE,))
    html_block('<div class="footer-note">© CAMERA CLICKS PHOTOGRAPHY</div>')


# ==========================================
# CAPTURE PAGE
# ==========================================
def metadata_fields(prefix: str) -> dict:
    title = st.text_input("Title", key=f"{prefix}_title", max_chars=120, placeholder="e.g. Golden hour over the lake")
    description = st.text_area("Description", key=f"{prefix}_desc", max_chars=1000, height=100)
    c1, c2 = st.columns(2)
    category = c1.selectbox("Category", CATEGORIES, key=f"{prefix}_cat")
    location = c2.text_input("Location", key=f"{prefix}_loc", max_chars=150, placeholder="City, place or landmark")
    c3, c4 = st.columns(2)
    tags = c3.text_input("Tags (comma separated)", key=f"{prefix}_tags", max_chars=200, placeholder="sunset, lake, calm")
    photo_date = c4.date_input("Date", value=date.today(), max_value=date.today(), key=f"{prefix}_date")
    return {"title": title, "description": description, "category": category,
            "location": location, "tags": tags, "photo_date": photo_date}


def resolve_title(title: str, fallback_name: str, index: int, total: int) -> str:
    title = clean_text(title, 120)
    if not title:
        stem = Path(fallback_name).stem.replace("_", " ").replace("-", " ")
        return clean_text(stem, 120) or "Untitled"
    return title if total == 1 else f"{title} {index + 1}"


def save_new_photos(items: list[tuple[str, bytes]], meta: dict) -> tuple[int, list[str]]:
    saved, problems = 0, []
    for i, (name, data) in enumerate(items):
        try:
            validate_upload(name, data)
            img = decode_image(data)
            title = resolve_title(meta["title"], name, i, len(items))
            filename = save_pil_image(img, title)
            new_id = add_photo(title, meta["description"], meta["category"], meta["location"], meta["tags"],
                               meta["photo_date"], filename, f"{PHOTOS_DIR.name}/{filename}")
            if new_id is None:
                with suppress(OSError):
                    (PHOTOS_DIR / filename).unlink()
                problems.append(f"{name}: could not be saved to the database.")
                continue
            saved += 1
        except ImageError as exc:
            problems.append(f"{name}: {exc}")
    return saved, problems


def finish_save(saved: int, problems: list[str], counter_key: str) -> None:
    if saved:
        flash(f"Saved {saved} photograph{'s' if saved != 1 else ''} to your gallery. 🎉", "success")
        for problem in problems:
            flash(problem, "warning")
        st.session_state[counter_key] += 1
        st.rerun()
    else:
        for problem in problems:
            st.error(problem)


def capture_page() -> None:
    show_flash()
    section_heading("Capture", "Capture a Moment", "Use your camera or upload photographs, then describe them.")
    tab_cam, tab_up = st.tabs(["📷 Use Camera", "⬆️ Upload Photos"])

    with tab_cam:
        st.info("Your browser will ask for camera permission. No camera, or access blocked? Use the Upload tab instead.")
        n = st.session_state["cam_n"]
        try:
            shot = st.camera_input("Take a photograph", key=f"cam_{n}")
        except Exception:
            shot = None
            st.warning("The camera is unavailable. Please use the Upload tab.")
        if shot is not None:
            data = shot.getvalue()
            with st.form(f"cam_form_{n}"):
                meta = metadata_fields(f"cam{n}")
                submitted = st.form_submit_button("💾 Save Photograph", type="primary")
            if submitted:
                saved, problems = save_new_photos([("Untitled capture.jpg", data)], meta)
                finish_save(saved, problems, "cam_n")

    with tab_up:
        n = st.session_state["up_n"]
        files = st.file_uploader("Choose photographs (JPG, JPEG, PNG, WEBP)", type=ALLOWED_EXTENSIONS,
                                 accept_multiple_files=True, key=f"up_{n}")
        valid_items: list[tuple[str, bytes]] = []
        if files:
            preview_cols = st.columns(4)
            for i, f in enumerate(files):
                try:
                    data = f.getvalue()
                    validate_upload(f.name, data)
                    img = decode_image(data)
                    valid_items.append((f.name, data))
                    if i < 8:
                        preview_cols[i % 4].image(img, caption=f.name, width=170)
                except ImageError as exc:
                    st.error(f"{f.name}: {exc}")
            if valid_items:
                st.caption(f"{len(valid_items)} valid photograph(s) ready. Leave the title empty to use file names.")
                with st.form(f"up_form_{n}"):
                    meta = metadata_fields(f"up{n}")
                    submitted = st.form_submit_button("💾 Save Photographs", type="primary")
                if submitted:
                    saved, problems = save_new_photos(valid_items, meta)
                    finish_save(saved, problems, "up_n")


# ==========================================
# GALLERY PAGE
# ==========================================
def render_photo_detail(photo: dict, all_photos: list[dict]) -> None:
    st.button("← Back to gallery", key="detail_back", on_click=cb_close_photo)
    left, right = st.columns([1.7, 1])
    uri = photo_uri(photo, 1600)
    with left:
        if uri:
            html_block(f'<div class="detail-frame"><img src="{uri}" alt="{esc(photo["title"])}"></div>')
        else:
            html_block('<div class="detail-frame"><div class="photo-missing">Image file not found on disk</div></div>')
    with right:
        fav_chip = '<span class="chip gold">❤️ Favorite</span>' if photo["is_favorite"] else ""
        tag_html = "".join(f'<span class="chip">#{esc(t)}</span>' for t in photo["tag_list"]) or '<span class="chip">No tags</span>'
        description = esc(photo["description"]).replace("\n", "<br>") or "No description added."
        html_block(
            f'<div class="detail-title">{esc(photo["title"])}</div>'
            f'<div class="chips"><span class="chip gold">{CATEGORY_EMOJI.get(photo["category"], "✨")} {esc(photo["category"])}</span>{fav_chip}</div>'
            f'<div class="detail-meta"><div><span>📍 Location</span>{esc(photo["location"]) or "Not specified"}</div>'
            f'<div><span>📅 Photo date</span>{fmt_date(photo["photo_date"])}</div>'
            f'<div><span>🕒 Added</span>{fmt_date(photo["created_at"])}</div></div>'
            f'<div class="detail-desc">{description}</div><div class="chips">{tag_html}</div>'
        )
        st.button("💔 Remove from favorites" if photo["is_favorite"] else "🤍 Add to favorites",
                  key="detail_fav", type="primary", on_click=cb_toggle_favorite, args=(photo["id"],))
        data = read_file_bytes(photo)
        if data:
            st.download_button("⬇️ Download photograph", data=data, file_name=f"{slugify(photo['title'])}.jpg",
                               mime="image/jpeg", key="detail_dl")
        else:
            st.warning("The image file is missing, so it cannot be downloaded.")
        st.button("✨ Open in Photo Editor", key="detail_edit", on_click=cb_edit_photo, args=(photo["id"],))

    with st.expander("✏️ Edit details"):
        with st.form(f"edit_form_{photo['id']}"):
            title = st.text_input("Title", value=photo["title"], max_chars=120)
            description = st.text_area("Description", value=photo["description"], max_chars=1000)
            c1, c2 = st.columns(2)
            category = c1.selectbox("Category", CATEGORIES,
                                    index=CATEGORIES.index(photo["category"]) if photo["category"] in CATEGORIES else len(CATEGORIES) - 1)
            location = c2.text_input("Location", value=photo["location"], max_chars=150)
            c3, c4 = st.columns(2)
            tags = c3.text_input("Tags (comma separated)", value=photo["tags"], max_chars=200)
            photo_date = c4.date_input("Date", value=photo["photo_date"] or date.today(), max_value=date.today())
            if st.form_submit_button("💾 Save changes", type="primary"):
                if not clean_text(title, 120):
                    st.error("Please enter a title.")
                elif update_photo(photo["id"], title=clean_text(title, 120), description=clean_text(description, 1000),
                                  category=category, location=clean_text(location, 150), tags=clean_tags(tags),
                                  photo_date=photo_date):
                    flash("Details updated.", "success")
                    st.rerun()
    with st.expander("🗑️ Delete photograph"):
        confirm = st.checkbox("I understand this permanently deletes the photograph.", key=f"del_confirm_{photo['id']}")
        if st.button("Delete permanently", key=f"del_btn_{photo['id']}"):
            if not confirm:
                st.warning("Please tick the confirmation box first.")
            elif delete_photo(photo["id"]):
                st.session_state["selected_photo_id"] = None
                flash("Photograph deleted.", "success")
                st.rerun()

    similar = similar_photos(photo, all_photos, 3)
    if similar:
        section_heading("Similar Photos", "You May Also Like", "Matched by colour and tone using scikit-learn.")
        render_photo_grid(similar, "sim", 3)


def gallery_page() -> None:
    photos = get_all_photos()
    show_flash()
    section_heading("Gallery", "The Collection", "Browse, search and filter every photograph.")

    selected_id = st.session_state.get("selected_photo_id")
    if selected_id is not None:
        photo = next((p for p in photos if p["id"] == selected_id), None)
        if photo:
            render_photo_detail(photo, photos)
            return
        st.session_state["selected_photo_id"] = None

    if not photos:
        empty_state("🖼️", "The gallery is empty", "Capture or upload your first photograph to get started.",
                    "📸 Add a photograph", PAGE_CAPTURE, "gal_empty")
        return

    locations = ["All locations"] + sorted({p["location"] for p in photos if p["location"]})
    all_tags = sorted({t for p in photos for t in p["tag_list"]})
    if st.session_state.get("g_location") not in locations:
        st.session_state["g_location"] = "All locations"
    st.session_state["g_tags"] = [t for t in st.session_state.get("g_tags", []) if t in all_tags]

    with st.expander("🎛️ Search & Filters", expanded=True):
        r1a, r1b = st.columns([3, 1])
        r1a.text_input("🔎 Search", key="g_query", placeholder="Title, description, category, location or tag…")
        r1b.selectbox("Sort by", SORT_OPTIONS, key="g_sort")
        r2a, r2b, r2c = st.columns(3)
        r2a.multiselect("Category", CATEGORIES, key="g_categories")
        r2b.selectbox("Location", locations, key="g_location")
        r2c.multiselect("Tags", all_tags, key="g_tags")
        r3a, r3b, r3c, r3d = st.columns([1, 2, 1, 1])
        use_date = r3a.toggle("Filter by date", key="g_use_date")
        date_range = None
        if use_date:
            dates = [p["photo_date"] for p in photos if p["photo_date"]] or [date.today()]
            picked = r3b.date_input("Date range", value=(min(dates), max(dates)), key="g_dates")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                date_range = (picked[0], picked[1])
            else:
                r3b.caption("Pick an end date to apply the range.")
        r3c.toggle("Favorites only", key="g_fav_only")
        r3d.button("Reset filters", key="g_reset", on_click=cb_reset_filters)

    s = st.session_state
    filtered = filter_photos(photos, s["g_query"], s["g_categories"], s["g_location"], s["g_tags"],
                             date_range, s["g_fav_only"], s["g_sort"])
    st.caption(f"Showing {len(filtered)} of {len(photos)} photographs")
    if not filtered:
        empty_state("🔍", "No photographs match", "Try a different search or reset the filters.")
        return

    pc1, pc2, _ = st.columns([1, 1, 3])
    per_page = pc1.selectbox("Photos per page", [9, 18, 36], key="g_per_page")
    total_pages = max(1, math.ceil(len(filtered) / per_page))
    if s["g_page"] > total_pages:
        s["g_page"] = total_pages
    page_no = pc2.number_input("Page", min_value=1, max_value=total_pages, step=1, key="g_page") if total_pages > 1 else 1
    start = (int(page_no) - 1) * per_page
    render_photo_grid(filtered[start:start + per_page], "gal", 3)


# ==========================================
# FAVORITES PAGE
# ==========================================
def favorites_page() -> None:
    photos = get_all_photos()
    show_flash()
    section_heading("Favorites", "Your Best Work", "The photographs you love the most.")
    favs = [p for p in photos if p["is_favorite"]]
    if not favs:
        empty_state("❤️", "No favorites yet", "Tap the heart on any photograph in the gallery to add it here.",
                    "🖼️ Browse the gallery", PAGE_GALLERY, "fav_empty")
        return
    render_photo_grid(favs, "fav", 3)
    recs = recommend_from_favorites(photos, favs, 3)
    if recs:
        section_heading("Recommended", "Because You Liked These",
                        "Photos with a similar colour palette to your favorites.")
        render_photo_grid(recs, "rec", 3)


# ==========================================
# PHOTO EDITOR
# ==========================================
def photo_editor_page() -> None:
    photos = get_all_photos()
    show_flash()
    section_heading("Photo Editor", "Edit Without Fear", "Your original photograph is never overwritten.")

    options = ["Gallery photo", "Upload a new image"] if photos else ["Upload a new image"]
    if st.session_state.get("ed_source") not in options:
        st.session_state["ed_source"] = options[0]
    source = st.radio("Image source", options, key="ed_source", horizontal=True)

    original: Optional[Image.Image] = None
    base_title, base_category, base_name = "Edited photo", "Other", "edited"
    if source == "Gallery photo":
        ids = [p["id"] for p in photos]
        labels = {p["id"]: f"{p['title']}  ·  {p['category']}" for p in photos}
        if st.session_state.get("ed_pick") not in ids:
            st.session_state["ed_pick"] = ids[0]
        pick = st.selectbox("Choose a photograph", ids, key="ed_pick", format_func=lambda i: labels.get(i, str(i)))
        chosen = next(p for p in photos if p["id"] == pick)
        original = load_pil(abs_path(chosen["image_path"]), EDITOR_WORK_SIDE)
        base_title, base_category, base_name = chosen["title"], chosen["category"], chosen["title"]
        if original is None:
            st.error("This image file is missing or corrupted.")
            return
    else:
        upload = st.file_uploader("Upload an image to edit", type=ALLOWED_EXTENSIONS, key="ed_upload")
        if upload is None:
            st.info("Upload an image to start editing.")
            return
        try:
            data = upload.getvalue()
            validate_upload(upload.name, data)
            original = decode_image(data)
            original.thumbnail((EDITOR_WORK_SIDE, EDITOR_WORK_SIDE), Image.Resampling.LANCZOS)
            base_name = Path(upload.name).stem
            base_title = base_name
        except ImageError as exc:
            st.error(str(exc))
            return

    st.markdown("#### Adjustments")
    a1, a2, a3, a4 = st.columns(4)
    brightness = a1.slider("Brightness", min_value=0.2, max_value=2.0, step=0.05, key="ed_brightness")
    contrast = a2.slider("Contrast", min_value=0.2, max_value=2.0, step=0.05, key="ed_contrast")
    saturation = a3.slider("Saturation", min_value=0.0, max_value=2.5, step=0.05, key="ed_saturation")
    sharpness = a4.slider("Sharpness", min_value=0.0, max_value=3.0, step=0.1, key="ed_sharpness")
    b1, b2, b3, b4 = st.columns(4)
    blur = b1.slider("Blur", min_value=0, max_value=20, step=1, key="ed_blur")
    rotation = b2.slider("Rotation (°)", min_value=-180, max_value=180, step=1, key="ed_rotation")
    resize_pct = b3.slider("Resize (%)", min_value=10, max_value=200, step=5, key="ed_resize")
    with b4:
        grayscale = st.toggle("Grayscale", key="ed_grayscale")
        st.button("↺ Reset adjustments", key="ed_reset", on_click=cb_reset_editor)

    try:
        edited = apply_edits(original, brightness, contrast, sharpness, saturation, grayscale, blur, rotation, resize_pct)
    except Exception as exc:  # defensive: never crash the page
        st.error(f"Could not apply the edits: {exc}")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        html_block('<div class="compare-label">Original Photo</div>'
                   f'<div class="detail-frame"><img src="{pil_to_data_uri(original, 1100)}"></div>')
        st.caption(f"{original.size[0]} × {original.size[1]} px")
    with col_b:
        html_block('<div class="compare-label">Edited Photo</div>'
                   f'<div class="detail-frame"><img src="{pil_to_data_uri(edited, 1100)}"></div>')
        st.caption(f"{edited.size[0]} × {edited.size[1]} px")

    st.markdown("#### Export")
    e1, e2, e3, e4 = st.columns(4)
    fmt = e1.selectbox("Download format", ["JPEG", "PNG"], key="ed_fmt")
    e2.download_button("⬇️ Download edited image", data=image_to_bytes(edited, fmt),
                       file_name=f"{slugify(base_name)}-edited.{'png' if fmt == 'PNG' else 'jpg'}",
                       mime="image/png" if fmt == "PNG" else "image/jpeg", key="ed_dl", type="primary")
    if e3.button("💾 Save copy to edited_photos", key="ed_save_copy"):
        try:
            name = save_pil_image(edited, f"edited-{base_name}", EDITED_DIR)
            st.success(f"Saved as {EDITED_DIR.name}/{name}")
        except ImageError as exc:
            st.error(str(exc))
    if e4.button("➕ Add edited version to gallery", key="ed_add_gallery"):
        try:
            name = save_pil_image(edited, f"edited-{base_name}", PHOTOS_DIR)
            new_id = add_photo(f"{base_title} (edited)", "Edited version created in the Photo Editor.", base_category,
                               "", "edited", date.today(), name, f"{PHOTOS_DIR.name}/{name}")
            if new_id:
                st.success("Added to your gallery as a new photograph. The original is untouched.")
        except ImageError as exc:
            st.error(str(exc))


# ==========================================
# ANALYTICS PAGE
# ==========================================
def show_chart(fig) -> None:
    if fig is None:
        st.info("Not enough data for this chart yet.")
        return
    st.pyplot(fig)
    plt.close(fig)


def analytics_page() -> None:
    photos = get_all_photos()
    section_heading("Analytics", "Your Photography in Numbers", "Insights from your collection.")
    if not photos:
        empty_state("📊", "No data yet", "Add some photographs and your dashboard will come alive.",
                    "📸 Add a photograph", PAGE_CAPTURE, "ana_empty")
        return
    df = photos_to_dataframe(photos)
    fav_count = int(df["is_favorite"].sum())
    latest = df["created_at"].max()
    html_block(
        '<div class="stat-grid">'
        f'<div class="stat-card"><div class="num">{len(df)}</div><div class="lbl">Total photographs</div></div>'
        f'<div class="stat-card"><div class="num">{df["category"].nunique()}</div><div class="lbl">Categories used</div></div>'
        f'<div class="stat-card"><div class="num">{fav_count}</div><div class="lbl">Favorites</div></div>'
        f'<div class="stat-card"><div class="num" style="font-size:1.6rem">{latest.strftime("%d %b %Y") if pd.notna(latest) else "—"}</div>'
        '<div class="lbl">Latest upload</div></div></div>'
    )
    c1, c2 = st.columns(2)
    with c1:
        show_chart(chart_by_category(df))
    with c2:
        show_chart(chart_by_month(df))
    c3, c4 = st.columns(2)
    with c3:
        show_chart(chart_by_location(df))
    with c4:
        show_chart(chart_favorites(df))

    section_heading("Recent", "Recent Photographs")
    render_photo_grid(photos[:4], "ana", 4)
    with st.expander("📋 Collection table"):
        table = df[["title", "category", "location", "photo_date", "is_favorite", "created_at"]].copy()
        table["photo_date"] = table["photo_date"].dt.date
        table.columns = ["Title", "Category", "Location", "Photo date", "Favorite", "Added"]
        st.dataframe(table)


# ==========================================
# ABOUT PAGE
# ==========================================
def about_page() -> None:
    section_heading("About", "Camera Clicks Photography", "Showcase. Organise. Create.")
    html_block(
        '<div class="glass"><p>Camera Clicks Photography is a project designed to showcase and manage photography '
        'collections. It combines a gallery-style portfolio with practical tools: camera capture, uploads, favorites, '
        'powerful search, a non-destructive photo editor and an analytics dashboard, all running locally on your computer '
        'from a single Python file.</p></div>'
    )
    section_heading("Technologies Used", "Built With")
    techs = [("🐍", "Python", "Core language"), ("🎈", "Streamlit", "Web interface"),
             ("🗄️", "SQLAlchemy", "ORM and models"), ("💾", "SQLite", "Local database"),
             ("👁️", "OpenCV", "Blur, grayscale, histograms"), ("🖼️", "Pillow", "Image loading and enhancing"),
             ("🐼", "Pandas", "Data handling"), ("🔢", "NumPy", "Numerical arrays"),
             ("📈", "Matplotlib", "Analytics charts"), ("🤖", "Scikit-learn", "Similar-photo recommendations")]
    cards = "".join(f'<div class="cat-card"><div class="emoji">{i}</div><div class="cname">{esc(n)}</div>'
                    f'<div class="ccount">{esc(d)}</div></div>' for i, n, d in techs)
    html_block(f'<div class="cat-grid">{cards}</div>')
    html_block('<div class="footer-note">CAMERA CLICKS PHOTOGRAPHY · DESIGNED FOR PHOTOGRAPHERS</div>')


# ==========================================
# MAIN APPLICATION
# ==========================================
def main() -> None:
    try:
        ensure_directories()
    except OSError as exc:
        st.error(f"Could not create the photo folders: {exc}")
        st.stop()
    try:
        get_session_factory()
    except SQLAlchemyError as exc:
        st.error(f"Could not open the database: {exc.__class__.__name__}")
        st.stop()

    inject_css()
    init_state()
    render_sidebar()

    page_functions = {
        PAGE_HOME: home_page, PAGE_CAPTURE: capture_page, PAGE_GALLERY: gallery_page,
        PAGE_FAVORITES: favorites_page, PAGE_EDITOR: photo_editor_page,
        PAGE_ANALYTICS: analytics_page, PAGE_ABOUT: about_page,
    }
    try:
        page_functions.get(st.session_state["nav_page"], home_page)()
    except Exception as exc:  # last-resort safety net so the app never shows a raw traceback
        st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()