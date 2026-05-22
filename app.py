"""
Review Miner v2 — AI Ad Copy Generator (Enterprise SaaS Edition)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from google_play_scraper import reviews as gplay_reviews, Sort, search as gplay_search
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import google.generativeai as genai
import json
import os
import io
import base64
from dotenv import load_dotenv
import re

load_dotenv()

def clean_app_name(raw_name: str) -> str:
    if not raw_name: return ""
    # Chops off anything after a colon, dash, or pipe to get the pure brand name
    return re.split(r'[:\-\|]', raw_name)[0].strip()

if "results" not in st.session_state:
    st.session_state.results = None
if "show_slider_help" not in st.session_state:
    st.session_state.show_slider_help = True

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Review Miner", page_icon="⛏️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Base Theme & Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Force Deep Black Canvas & Viewport Glow */
.stApp {
    background-color: #0e1117;
    background: linear-gradient(135deg, #002159, #070029, #0057ff) !important;
    box-shadow: inset 0 0 100px rgba(0, 210, 255, 0.05);
    overflow-x: hidden !important; /* Prevents horizontal scroll from large backgrounds */
}
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stSidebar"] {
    background-color: #040508 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Hybrid Layout Constraints */
.block-container {
    max-width: 1040px !important;
    padding-top: 0rem !important;
    padding-bottom: 5rem !important;
}

/* Minimalist Flat Cards for Containers */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background: #06070B !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important; 
    box-shadow: none !important;
    transition: all 0.2s ease;
}
div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Transparent Primary Button with Gradient Border */
button[kind="primary"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 1;
}
button[kind="primary"]::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 9999px;
    padding: 3px; /* Controls border thickness */
    background: linear-gradient(90deg, #8c52ff, #ff914d);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    z-index: -1;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 25px rgba(140, 82, 255, 0.4) !important;
    transform: translateY(-2px);
}
button[kind="primary"]:active {
    transform: scale(0.98);
}

/* Dynamic Tab Styling */
div[data-testid="stTabs"] { gap: 0 !important; }
div[data-baseweb="tab-list"] { border-bottom: none !important; }
div[data-baseweb="tab-border"] { display: none !important; }
div[data-testid="stTabs"] > div[data-baseweb="tab-list"] ~ div[data-testid="stVerticalBlock"] {
    padding-top: 1rem !important;
}

/* 1. BASE TAB STYLING (Unselected State) */
button[data-baseweb="tab"] {
    background-color: transparent !important; 
    border-radius: 9999px !important;
    
    /* FONT & TEXT SIZE */
    font-family: 'Poppins', sans-serif !important; 
    font-size: 2rem !important; 
    font-weight: 600 !important;
    color: #8C92A6 !important; 
    
    /* BUTTON SIZE (Padding) */
    padding: 10px 28px !important; 
    margin-right: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease;
}

/* 2. HOVER STATE (When you mouse over a tab) */
button[data-baseweb="tab"]:hover {
    border-color: rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important; 
    background-color: rgba(255, 255, 255, 0.05) !important; 
}

/* 3. ACTIVE/SELECTED STATE COLORS (Individual Tabs) */

/* Tab 1: 📊 Health & Insight */
button[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    border-color: #39FF14 !important; 
    color: #39FF14 !important; 
    box-shadow: 0 0 15px rgba(57, 255, 20, 0.2) !important;
}

/* Tab 2: 🧠 Strategy */
button[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    border-color: #B026FF !important; 
    color: #B026FF !important; 
    box-shadow: 0 0 15px rgba(176, 38, 255, 0.2) !important;
}

/* Tab 3: ✍️ Content & Captions */
button[data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
    border-color: #00D2FF !important; 
    color: #00D2FF !important; 
    box-shadow: 0 0 15px rgba(0, 210, 255, 0.2) !important;
}

div[data-baseweb="tab-highlight"] { display: none !important; }

/* Metrics Styling */
div[data-testid="stMetricValue"] {
    font-weight: 800 !important; color: #ffffff !important; font-size: 2.5rem !important; letter-spacing: -1px;
}
div[data-testid="stMetricLabel"] {
    color: #8C92A6 !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 1px; font-size: 0.75rem !important;
}

/* Custom Input Field Styling */
div[data-baseweb="input"] > div, div[data-baseweb="input"] {
    background-color: #8c89d9 !important;
    border-radius: 12px !important;
    border: none !important;
}
div[data-baseweb="input"] {
    border: 1px solid rgba(255,255,255,0.3) !important;
}
div[data-baseweb="input"] input {
    color: white !important;
    background-color: transparent !important;
    -webkit-text-fill-color: white !important;
}
div[data-baseweb="input"] input::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
    -webkit-text-fill-color: rgba(255, 255, 255, 0.7) !important;
}

/* Selectbox Dropdowns */
.stSelectbox > div > div > div {
    background: #8c89d9 !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
}

/* 1. Gradient for the Slider Track Line ONLY */
div[data-baseweb="slider"] > div > div > div:nth-child(1) {
    background: linear-gradient(90deg, #ff66c4, #ffde59) !important;
}

/* 2. Pure Transparent Background & White Text for Min (4) and Max (10) */
div[data-testid="stSliderTickBarMin"], 
div[data-testid="stSliderTickBarMax"],
div[data-testid="stSliderTickBarMin"] *, 
div[data-testid="stSliderTickBarMax"] * {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important; 
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important; 
    -webkit-text-fill-color: #ffffff !important;
}

/* Main Text & Headings */
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
p, li { color: #8C92A6 !important; }

/* Component Headings — Enlarged to 2rem */
.platform-header {
    font-size: 2rem !important; /* Enlarged from 1.2rem */
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 0.5rem;
}

/* Extracted Themes Table Styling */
table { width: 100% !important; border-collapse: collapse !important; }
th {
    background-color: rgba(0, 210, 255, 0.08) !important; color: #00D2FF !important;
    font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.5px;
    font-size: 0.8rem; padding: 12px 16px !important; border-bottom: 1px solid rgba(0, 210, 255, 0.3) !important;
}
td {
    padding: 12px 16px !important; border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #E2E8F0 !important; font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ── Cached data functions ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def search_gplay(query: str) -> list[dict]:
    try:
        results = gplay_search(query, n_hits=30, lang="en", country="my")
    except Exception:
        return []
    
    valid_apps = [{"title": r["title"], "appId": r["appId"]} for r in results if r.get("appId")]
    import re
    query_lower = query.lower().strip()
    
    def relevance_score(app):
        title = app["title"].lower()
        if title == query_lower or re.match(rf"^{re.escape(query_lower)}\b", title): return 0           
        if title.startswith(query_lower): return 1
        if query_lower in title: return 2
        return 3
        
    valid_apps.sort(key=relevance_score)
    return valid_apps[:15]

@st.cache_data(show_spinner=False)
def scrape_gplay(app_id: str, count: int) -> list[dict]:
    fetch_amount = count * 3 
    result, _ = gplay_reviews(app_id, lang="en", country="my", sort=Sort.NEWEST, count=fetch_amount)
    valid_reviews = [{"text": r["content"], "score": r["score"], "thumbs_up": r.get("thumbsUpCount", 0), "date": str(r.get("at", ""))[:10], "platform": "Google Play"}
            for r in result if r.get("content") and len(r["content"].strip()) > 20]
    return valid_reviews[:count]

@st.cache_data(show_spinner=False)
def get_appstore_name(app_id: str) -> str:
    try:
        resp = requests.get(f"https://itunes.apple.com/lookup?id={app_id}", timeout=5)
        data = resp.json()
        if data.get("resultCount", 0) > 0:
            return data["results"][0]["trackName"]
    except Exception:
        pass
    return f"App Store App ({app_id})"

@st.cache_data(show_spinner=False)
def search_appstore(query: str) -> list[dict]:
    try:
        resp = requests.get(f"https://itunes.apple.com/search?term={query}&entity=software&limit=10&country=my", timeout=5)
        data = resp.json()
        return [{"title": r["trackName"], "appId": str(r["trackId"])} for r in data.get("results", [])]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def scrape_appstore(app_id: str, count: int) -> list[dict]:
    import requests
    import streamlit as st
    try:
        api_key = st.secrets.get("SERPAPI_KEY", "")
        if not api_key:
            st.error("SERPAPI_KEY is missing in secrets.toml. Please add it to your Streamlit secrets.")
            return []
        reviews = []
        page = 1
        while len(reviews) < count:
            url = f"https://serpapi.com/search.json?engine=apple_reviews&product_id={app_id}&api_key={api_key}&page={page}&sort=mostrecent&country=my"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                st.error(f"SerpApi Error: Failed to fetch reviews (Status {resp.status_code})")
                break
            data = resp.json()
            entries = data.get("reviews", [])
            if not entries:
                break
            for entry in entries:
                text = entry.get("text", "")
                score = entry.get("rating", 3)
                date_str = entry.get("date", "")
                if len(text.strip()) > 20:
                    reviews.append({
                        "text": text,
                        "score": score,
                        "thumbs_up": 0,
                        "date": date_str,
                        "platform": "App Store"
                    })
                if len(reviews) >= count:
                    break
            page += 1
        return reviews[:count]
    except Exception as e:
        print(f"App Store SerpApi scrape error: {e}")
        return []

def analyse_reviews(reviews_data: list[dict], n_clusters: int = 6) -> tuple:
    analyser = SentimentIntensityAnalyzer()
    for r in reviews_data:
        compound = analyser.polarity_scores(r["text"])["compound"]
        r["sentiment"] = round(compound, 4)
        r["sentiment_label"] = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    
    texts = [r["text"] for r in reviews_data]
    safe_k = min(n_clusters, max(2, len(texts) // 10))
    vec = TfidfVectorizer(max_features=800, stop_words="english", ngram_range=(1, 2), min_df=2)
    
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        for r in reviews_data: r["cluster"] = 0
        return reviews_data, {0: {"keywords": ["insufficient text data"], "reviews": reviews_data, "size": len(reviews_data), "avg_sentiment": 0.0}}

    km = KMeans(n_clusters=safe_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    for i, r in enumerate(reviews_data): r["cluster"] = int(labels[i])
    feature_names = vec.get_feature_names_out()
    themes = {}
    for i in range(safe_k):
        cluster_reviews = [reviews_data[j] for j, l in enumerate(labels) if l == i]
        centroid = km.cluster_centers_[i]
        top_idx = centroid.argsort()[-8:][::-1]
        keywords = [feature_names[idx] for idx in top_idx]
        avg_sentiment = sum(r["sentiment"] for r in cluster_reviews) / len(cluster_reviews) if cluster_reviews else 0.0
        themes[i] = {"keywords": keywords, "reviews": cluster_reviews, "size": len(cluster_reviews), "avg_sentiment": round(avg_sentiment, 3)}
    return reviews_data, themes

@st.cache_data(show_spinner=False)
def generate_copy(product_name: str, themes_summary: str, neg_sample: str, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    output_spec = """{
  "health_and_insight": {
    "pain_and_praise": {
      "praise_insight": "Insight on what to brag about based on top positive clusters",
      "pain_insight": "Insight on what friction to address immediately based on top negative clusters"
    },
    "feature_request_backlog": ["Feature 1: ...", "Feature 2: ...", "Feature 3: ..."],
    "friction_questions": ["Question 1", "Question 2"]
  },
  "marketing_strategy": {
    "core_positioning": "Define the core positioning angle. DO NOT write ad copy.",
    "aso_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "competitor_counter_angle": "Define the exact angle to beat rivals. DO NOT write ad copy.",
    "retention_objection_handler": "Identify the biggest user fear/friction and neutralize it. DO NOT write ad copy."
  },
  "content_and_captions": {
    "app_store_subtitle": "Max 30 chars",
    "whats_new_release_notes": "Directly addressing the scraped bugs/complaints",
    "x_thread": ["Hook...", "Tweet 2...", "Tweet 3..."],
    "reddit_post": {"title": "...", "body": "..."},
    "tiktok_ugc_hook": "Indie dev 'Look at this app I just coded' hook",
    "product_hunt_comment": "The launch pitch"
  }
}"""
    tone_note = """Dynamic and highly authentic. Write at an 8th-grade reading level.
Keep sentences punchy and avoid AI jargon."""
    prompt = f"Product: {product_name}\nThemes: {themes_summary}\nBad Reviews: {neg_sample}\nTone: {tone_note}\nReturn ONLY valid JSON matching this exact structure:\n{output_spec}"
    
    # Use the fastest, most reliable text modelgit add .
    fixed_model_name = "gemini-flash-lite-latest"
    model = genai.GenerativeModel(fixed_model_name)
    response = model.generate_content(prompt)
    
    raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw_text)
# Secure API Key Management
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to run analysis.")
    if not api_key:
        st.warning("Please configure your API key in `.streamlit/secrets.toml` or paste it above.")

# ── Load Assets ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_image_b64(filename):
    import base64
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "static")
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

bg1 = get_image_b64("background-1.png")
bg2 = get_image_b64("background-2.png")
bg3 = get_image_b64("background-3.png")
bg4 = get_image_b64("background-4.png")
bg_circle = get_image_b64("background-circle.png")
bubble_img = get_image_b64("bubble.png")
logo_img = get_image_b64("logo.png")
laptop_img = get_image_b64("laptop.png")
gplay_img = get_image_b64("googleplay.png")
apple_img = get_image_b64("appleapp.png")

# ── Landing screen ────────────────────────────────────────
landing_html = f"""
<style>
/* Landing Container (Overflow visible so large waves bleed into lower sections) */
.custom-landing {{
    position: relative;
    width: 100vw;
    left: 50%;
    transform: translateX(-50%);
    margin-top: -3rem; /* Offset Streamlit Header */
    padding-top: 2rem;
    padding-bottom: 0rem;
    overflow: visible; 
    color: white;
    text-align: center;
    font-family: 'Inter', sans-serif;
}}

/* Massive, Edge-Bleeding Backgrounds */
.bg-layer {{
    position: absolute;
    pointer-events: none;
    z-index: -2;
    margin: 0;
}}
.bg-top-left {{ top: -7vh; left: -60vw; width: 100vw; transform: rotate(2deg) scale(1.3);mix-blend-mode: screen; 
    opacity: 0.9; }}
.bg-top-right {{ top: 40vh; right: -25vw; width: 100vw;transform: rotate(-175deg) scale(1.3);mix-blend-mode: screen; opacity: 0.9; }}
.bg-bottom-right {{ bottom: -10vh; right: 50vw; width: 110vw; transform: rotate(2deg) scale(1.8);mix-blend-mode: opacity: 0.8; }}
.bg-bottom-left {{ bottom: -35vh; left: 40vw; width: 110vw; transform: rotate(2deg) scale(1.3); mix-blend-mode: screen; opacity: 0.8; }}

/* Inner Content Alignment */
.landing-inner {{
    position: relative;
    z-index: 1;
    max-width: 1040px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.hero-logo {{
    width: 600px;
    max-width: 90%;
    height: auto;
    object-fit: contain;
    margin-bottom: -3px;
    filter: drop-shadow(0 0 40px rgba(160, 80, 255, 0.4));
}}

.hero-title {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800;
    font-size: 4.5rem;
    letter-spacing: -2px;
    margin: 0 0 1.5rem 0;
    color: white;
    line-height: 1.1;
}}

.hero-subtitle {{
    color: #E2E8F0;
    font-size: 1.6rem;
    max-width: 780px;
    margin: 0 auto 5rem auto;
    line-height: 1.6;
    letter-spacing: -0.2px;
}}

/* Tightened Performance Insights */
.metrics-title {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800;
    font-size: 3.5rem;
    color: #fabdff !important;
    margin-bottom: 0rem;
    letter-spacing: 1px;
}}

.metrics-subtitle {{
    color: #E2E8F0;
    font-size: 1.5rem;
    margin-bottom: 5rem;
    line-height: 1.3;
}}

.metrics-container {{
    display: flex;
    justify-content: center;
    gap: 6rem;
    margin-bottom: 7rem;
}}

.metric-item {{ text-align: center; }}

.metric-label {{
    background: linear-gradient(180deg, #ffde59, #ff914d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text; /* Standard fallback */
    font-size: 2rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: -0.8rem;
}}

.metric-value {{
    font-size: 4.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: white;
}}

.how-it-works-title {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800;
    font-size: 3rem;
    color: #fabdff !important;
    margin-bottom: -0.9rem;
    letter-spacing: 1px;
}}

/* Enlarged & Tightly Spaced Bubbles */
.bubbles-wrapper {{
    position: relative;
    width: 100%;
    height: 850px;
    margin: 0.8rem auto 10rem auto;
}}

.bubble {{
    position: absolute;
    width: 480px;
    height: 480px;
    border-radius: 50%;
    background-image: url('{bubble_img}');
    background-size: cover;
    background-position: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 4rem;
    text-align: center;
    transition: transform 0.3s ease;
}}

.bubble:hover {{ transform: translateY(-10px) scale(1.02); }}

.b1 {{ top: 0%; left: 6%; }}
.b2 {{ top: 30%; right: 6%; z-index: 2; }}
.b3 {{ top: 58%; left: 6%; z-index: 3; }}

.bubble h3 {{
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 1rem 0;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    line-height: 1.2;
}}

.bubble p {{
    font-size: 1.2rem;
    color: white;
    margin: 0;
    line-height: 1.5;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}}
</style>

<div class="custom-landing">
    <img src="{bg1}" class="bg-layer bg-top-left">
    <img src="{bg2}" class="bg-layer bg-top-right">
    <img src="{bg3}" class="bg-layer bg-bottom-right">
    <img src="{bg4}" class="bg-layer bg-bottom-left">

    <div class="landing-inner">
        <img src="{logo_img}" class="hero-logo">
        <h1 class="hero-title">Review Miner</h1>
        <p class="hero-subtitle">Transform app feedback into actionable product strategy and high-converting assets.</p>

        <div class="metrics-title">Performance Insights</div>
        <p class="metrics-subtitle">Direct-response analysis complete in < 60 seconds.</p>

        <div class="metrics-container">
            <div class="metric-item">
                <div class="metric-label">COST REDUCTION</div>
                <div class="metric-value">95%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">INSIGHT DEPTH</div>
                <div class="metric-value">HIGH</div>
            </div>
        </div>

        <div class="how-it-works-title">How Review Miner Works</div>

        <div class="bubbles-wrapper">
            <div class="bubble b1">
                <h3>Intelligent Data<br>Extraction</h3>
                <p>Enter a target app to instantly aggregate live, unfiltered user reviews from Google Play or the Apple App Store.</p>
            </div>
            <div class="bubble b2">
                <h3>NLP Health &<br>Insight Analysis</h3>
                <p>The system utilizes VADER sentiment scoring and TF-IDF + KMeans clustering to distill unstructured feedback into exact pain points, praise, and feature requests.</p>
            </div>
            <div class="bubble b3">
                <h3>Marketing Strategy<br>& Ad Copy</h3>
                <p>The AI synthesizes the data into a 4-pillar marketing strategy and generates highly-converting, founder-led ad copy.</p>
            </div>
        </div>
    </div>
</div>
"""

st.html(landing_html)

# ── App Extraction Core ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'''
        <div style='display: flex; justify-content: center; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem;'>
            <img src='{gplay_img}' style='height: 80px; width: auto; object-fit: contain; transform: scale(3);'>
        </div>
    ''', unsafe_allow_html=True)
    
    search_q = st.text_input("Search for an App", placeholder="e.g. Spotify, Uber", key="gplay_search_bar")
        
    gplay_id, gplay_name = "", ""
    if search_q:
        with st.spinner("Searching..."):
            hits = search_gplay(search_q)
        if hits:
            opts = {f"{h['title']}  ({h['appId']})": h for h in hits}
            chosen_key = st.selectbox("Select app", list(opts.keys()))
            gplay_id, gplay_name = opts[chosen_key]["appId"], opts[chosen_key]["title"]
            # st.code has been removed from here
        else:
            st.error("Misspelling / Invalid Brand")
    
    # The value=gplay_id parameter automatically fills this box with the dropdown selection
    manual = st.text_input("Or paste App ID directly", placeholder="com.spotify.music", value=gplay_id)
    # Only override if the user actually typed a NEW manual ID
    if manual.strip() and manual.strip() != gplay_id: 
        gplay_id = manual.strip()
        gplay_name = manual.strip().split(".")[-1].capitalize()
    
    gplay_count_input = st.number_input("Play Store reviews to fetch", min_value=5, max_value=500, value=None, placeholder="Max 500", step=1)
    gplay_count = gplay_count_input if gplay_count_input is not None else 50

with col2:
    st.markdown(f'''
        <div style='display: flex; justify-content: center; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem;'>
            <img src='{apple_img}' style='height: 80px; width: auto; object-fit: contain; transform: scale(3.4);'>
        </div>
    ''', unsafe_allow_html=True)
    search_ios_q = st.text_input("Search App Store", placeholder="e.g. Spotify, Uber", key="apple_search_bar")
    ios_id, ios_name = "", ""
    if search_ios_q:
        with st.spinner("Searching..."):
            ios_hits = search_appstore(search_ios_q)
        if ios_hits:
            opts = {f"{h['title']}  ({h['appId']})": h for h in ios_hits}
            chosen_key = st.selectbox("Select iOS app", list(opts.keys()))
            ios_id = opts[chosen_key]["appId"]
            ios_name = opts[chosen_key]["title"] # We now capture the name!
        else:
            st.error("Misspelling / Invalid Brand")
            
    # The value=ios_id parameter automatically fills this box with the dropdown selection
    manual_ios = st.text_input("Or paste numeric ID directly", placeholder="324684580", value=ios_id)
    # Only override if the user actually typed a NEW manual ID
    if manual_ios.strip() and manual_ios.strip() != ios_id: 
        ios_id = manual_ios.strip()
        ios_name = f"App Store App ({ios_id})" # Fallback if manual paste
    
    ios_count_input = st.number_input("App Store reviews to fetch", min_value=5, max_value=500, value=None, placeholder="Max 500", step=1)
    ios_count = ios_count_input if ios_count_input is not None else 50
    
st.divider()

# 1. Big, bold Analysis Settings heading
st.markdown("<div class='platform-header'>Analysis Settings</div>", unsafe_allow_html=True)

# 2. Normal text line sitting cleanly above the slider
st.markdown("<p style='color: white; font-size: 1rem; margin-bottom: 2rem;'> ✽ If the generated themes look too similar to each other, decrease the clusters.</p>", unsafe_allow_html=True)

# 3. The theme cluster slider without any built-in question marks
n_clusters = st.slider("Theme clusters (Number of pain/praise topics to extract)", 4, 10, 6)

st.divider()

can_scrape_gplay = bool(gplay_id and api_key)
can_scrape_ios = bool(ios_id and api_key)

if st.button("Mine & Generate Copy", type="primary", width="stretch", disabled=not (can_scrape_gplay or can_scrape_ios)):
    raw = []
    app_names = []
    warnings_to_show = []  # 1. Create a list to hold store-specific warnings
    
    if can_scrape_gplay:
        with st.spinner(f"Fetching Google Play reviews..."):
            gplay_data = scrape_gplay(gplay_id, gplay_count)
            raw.extend(gplay_data)
            
            # 2. Check Google Play specifically
            gplay_fetched = len(gplay_data)
            if gplay_fetched < gplay_count:
                warnings_to_show.append(f"You requested {gplay_count} reviews from Google Play, but there are only {gplay_fetched} reviews in the store. Proceeding with analysis using these {gplay_fetched} reviews.")
                
            if gplay_name: app_names.append(gplay_name)
            
    if can_scrape_ios:
        with st.spinner(f"Fetching App Store reviews..."):
            ios_data = scrape_appstore(ios_id.strip(), ios_count)
            raw.extend(ios_data)
            
            # 3. Check App Store specifically
            ios_fetched = len(ios_data)
            if ios_fetched < ios_count:
                warnings_to_show.append(f"You requested {ios_count} reviews from the App Store, but there are only {ios_fetched} reviews in the store. Proceeding with analysis using these {ios_fetched} reviews.")
            
            final_ios_name = ios_name if ios_name else get_appstore_name(ios_id.strip())
            
            if final_ios_name:
                clean_new = clean_app_name(final_ios_name).lower()
                if not any(clean_app_name(existing).lower() == clean_new for existing in app_names):
                    app_names.append(final_ios_name)
            
    # 4. Final check and display warnings
    total_fetched = len(raw)
    
    if total_fetched >= 10:
        # Display any warnings we collected for the specific stores
        for warning_msg in warnings_to_show:
            st.warning(f"⚠️ **Notice:** {warning_msg}")
            
        combined_name = " & ".join(app_names) if app_names else "Result"
        st.session_state.results = {"raw": raw, "name": combined_name}
    else: 
        st.error(f"Too few reviews returned across platforms (Found {total_fetched}). Need at least 10.")
            
    # 2. Check for the shortfall warning
    total_fetched = len(raw)
    
    if total_fetched >= 10:
        if total_fetched < total_requested:
            # 3. Pop up the warning if there are fewer reviews than requested
            st.warning(f"⚠️ **Notice:** You requested {total_requested} reviews, but we could only fetch {total_fetched} available in the store. Proceeding with analysis using these {total_fetched} reviews.")
            
        combined_name = " & ".join(app_names) if app_names else "Result"
        st.session_state.results = {"raw": raw, "name": combined_name}
    else: 
        # 4. Keep the original error if it's completely dead (under 10)
        st.error(f"Too few reviews returned across platforms (Found {total_fetched}). Need at least 10.")

# ── Shared pipeline ───────────────────────────────────────────────────────────
if st.session_state.results is not None:
    data = st.session_state.results
    with st.spinner("Running NLP analysis…"): reviews_data, themes = analyse_reviews(data["raw"], n_clusters)
    df = pd.DataFrame(reviews_data)
    
    neg_reviews = sorted([r for r in reviews_data if r["sentiment_label"] == "negative"], key=lambda x: x["thumbs_up"], reverse=True)
    themes_summary = "\n".join(f"Theme {i+1}: {', '.join(t['keywords'][:5])}" for i, t in themes.items())
    neg_sample = "\n".join(f'- "{r["text"][:200]}"' for r in neg_reviews[:6])

    with st.spinner("Generating copy with Gemini 3.1 Flash-Lite…"):
        try:
            copy_output = generate_copy(data["name"], themes_summary, neg_sample, api_key)
        except json.JSONDecodeError:
            st.error("Gemini returned malformed JSON. Try running again.")
            st.stop()
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            st.stop()

    st.divider()
    st.markdown(f"## {data['name']}")
    
    res1, res2, res3 = st.tabs(["Health & Insight", "Strategy", "Content & Captions"])

    with res1:
        st.markdown("### 1. The Executive Snapshot")
        platforms_present = df['platform'].unique()
        for platform in platforms_present:
            pdf = df[df['platform'] == platform]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{platform} Reviews", f"{len(pdf):,}")
            m2.metric("Positive 🟩", f"{len(pdf[pdf.sentiment_label == 'positive']) / len(pdf) * 100:.0f}%")
            m3.metric("Negative 🟥", f"{len(pdf[pdf.sentiment_label == 'negative']) / len(pdf) * 100:.0f}%")
            m4.metric("Avg rating", f"{pdf['score'].mean():.1f} ★")
            
        st.divider()
        st.markdown("### 2. The 'Pain & Praise' Matrix")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Sentiment")
            st.plotly_chart(px.pie(df, names="sentiment_label", color="sentiment_label", color_discrete_map={"positive": "#4ade80", "negative": "#f87171", "neutral": "#94a3b8"}, hole=0.45), width="stretch")
        with right:
            st.markdown("#### Stars")
            st.plotly_chart(px.bar(df["score"].value_counts().sort_index().reset_index(), x="score", y="count"), width="stretch")
        
        st.markdown("#### Extracted Themes")
        st.table(pd.DataFrame([{"Theme": f"Theme {i+1}", "Keywords": " · ".join(t["keywords"][:5]), "Reviews": t["size"], "Tone": "🟢" if t["avg_sentiment"] > 0 else "🔴"} for i, t in themes.items()]))
        
        hi = copy_output.get("health_and_insight", {})
        pain_praise = hi.get("pain_and_praise", {})
        
        if pain_praise:
            praise_text = pain_praise.get('praise_insight', '').replace('\n', '<br>')
            pain_text = pain_praise.get('pain_insight', '').replace('\n', '<br>')
            html_str = f"""
            <div style="display: flex; gap: 1.5rem; align-items: stretch; margin-bottom: 1.5rem; margin-top: 1rem;">
                <div style="flex: 1; background: #06070B; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 1.5rem; transition: all 0.2s ease;" onmouseover="this.style.borderColor='rgba(255,255,255,0.2)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.08)'">
                    <div style="color: #ffffff; font-weight: 700; margin-top: 0; margin-bottom: 0.75rem; font-family: 'Inter', sans-serif;">🟢 Praise Insight <span style="color: #8C92A6; font-size: 0.8em; font-weight: 500;">(Double down on)</span></div>
                    <div style="color: #8C92A6; margin-bottom: 0; font-family: 'Inter', sans-serif; font-size: 1rem; line-height: 1.6;">{praise_text}</div>
                </div>
                <div style="flex: 1; background: #06070B; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 1.5rem; transition: all 0.2s ease;" onmouseover="this.style.borderColor='rgba(255,255,255,0.2)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.08)'">
                    <div style="color: #ffffff; font-weight: 700; margin-top: 0; margin-bottom: 0.75rem; font-family: 'Inter', sans-serif;">🔴 Pain Insight <span style="color: #8C92A6; font-size: 0.8em; font-weight: 500;">(Fix immediately)</span></div>
                    <div style="color: #8C92A6; margin-bottom: 0; font-family: 'Inter', sans-serif; font-size: 1rem; line-height: 1.6;">{pain_text}</div>
                </div>
            </div>
            """
            st.markdown(html_str, unsafe_allow_html=True)
                
        st.markdown("### 3. The Feature Request Backlog")
        with st.container(border=True):
            backlog = hi.get("feature_request_backlog", [])
            if isinstance(backlog, list) and backlog:
                for req in backlog:
                    st.markdown(f"{req}") # Numbering removed, relies entirely on AI output
            else:
                st.markdown(str(backlog) if backlog else "No feature requests identified.")
            
        st.markdown("### 4. The 'Friction' Questions")
        with st.container(border=True):
            for q in hi.get("friction_questions", []):
                st.markdown(f"- {q}")
        
        st.divider()
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;'>
            <h3 style='color: #E2E8F0; margin-bottom: 0.2rem; font-weight: 700;'>Raw Review Data</h3>
            <div style='color: #8C92A6; font-size: 0.95rem; opacity: 0.9;'>Unfiltered voice of customer dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
        display_df = df[["platform", "score", "sentiment_label", "date", "text"]].copy()
        display_df.columns = ["Platform", "Rating", "Sentiment", "Date", "Review Text"]
        display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%d %b %Y")
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            hide_index=True, 
            height=400,
            column_config={
                "Platform": st.column_config.TextColumn(width="medium"),
                "Rating": st.column_config.NumberColumn(width="small"),
                "Sentiment": st.column_config.TextColumn(width="small"),
                "Date": st.column_config.TextColumn(width="small"),
                "Review Text": st.column_config.TextColumn(width="large")
            }
        )

    with res2:
        st.markdown("### Marketing Strategy")
        ms = copy_output.get("marketing_strategy", {})
        
        with st.container(border=True):
            st.markdown("##### 🎯 Core App Store Positioning")
            st.markdown(ms.get("core_positioning", ""))
            
        with st.container(border=True):
            st.markdown("##### 🔑 ASO Keywords")
            st.markdown(" · ".join([f"`{k}`" for k in ms.get("aso_keywords", [])]))
            
        with st.container(border=True):
            st.markdown("##### Competitor Counter-Angle")
            st.markdown(ms.get("competitor_counter_angle", ""))
            
        with st.container(border=True):
            st.markdown("##### Retention/Trust Objection Handler")
            st.markdown(ms.get("retention_objection_handler", ""))

    with res3:
        cc = copy_output.get("content_and_captions", {})
        
        with st.container(border=True):
            st.markdown("##### App Store Subtitle (Max 30 chars)")
            st.code(cc.get("app_store_subtitle", ""))
            
        with st.container(border=True):
            st.markdown("##### 'What's New' Release Notes")
            st.markdown(cc.get("whats_new_release_notes", ""))
            
        with st.container(border=True):
            st.markdown("##### X (Twitter) 'Build in Public' Thread")
            for i, t in enumerate(cc.get("x_thread", [])):
                st.markdown(f"**Tweet {i+1}:** {t}")
                
        with st.container(border=True):
            st.markdown("##### Targeted Reddit Post")
            reddit = cc.get("reddit_post", {})
            st.markdown(f"**Title:** {reddit.get('title', '')}")
            st.markdown(reddit.get("body", ""))
            
        with st.container(border=True):
            st.markdown("##### TikTok/Reels 'UGC' Hook")
            st.markdown(cc.get("tiktok_ugc_hook", ""))
            
        with st.container(border=True):
            st.markdown("##### Product Hunt Maker Comment")
            st.markdown(cc.get("product_hunt_comment", ""))
        
        st.divider()
        st.markdown("### API & Workflow Export")
        st.markdown("<p style='font-size: 1.1rem; color: #ffffff; margin-bottom: 1rem;'>Are you a developer? Export this entire strategic analysis as a structured JSON payload to instantly plug into Zapier, Notion, or your custom automated posting workflows.</p>", 
        unsafe_allow_html=True)
        
        st.download_button(
            label=" Export Action Plan (JSON)", 
            data=json.dumps(copy_output, indent=2), 
            file_name="review_miner_strategy.json", 
            mime="application/json",
            type="primary"
        )

else:
    st.info("Ready for extraction. Enter target apps above and click 'Mine & generate copy' to begin.")

st.divider()
st.caption("© 2026 Review Miner AI.")