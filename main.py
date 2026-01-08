# =========================
# IMPORTS
# =========================
import requests
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
import wikipediaapi
from openai import OpenAI
from dotenv import load_dotenv

import os
load_dotenv()

# =========================
# CONFIG
# =========================
YOUTUBE_API_KEY = "AIzaSyBPfdAAD9NTIluCAXmV3IiQFJxMjkplGVM"
TMDB_API_KEY = "1f929715ed395b7c8c435a3ea8dcb8c4"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# GOOGLE BOOKS
# =========================
@st.cache_data(show_spinner=False)
def fetch_books(query, max_results=10):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": max_results}
    data = requests.get(url, params=params).json()

    return [
        {
            "title": item.get("volumeInfo", {}).get("title", ""),
            "description": item.get("volumeInfo", {}).get("description", ""),
            "link": item.get("volumeInfo", {}).get("previewLink", "")
        }
        for item in data.get("items", [])
    ]

# =========================
# YOUTUBE
# =========================
@st.cache_data(show_spinner=False)
def fetch_youtube(query, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    response = youtube.search().list(
        q=query, part="snippet", type="video", maxResults=max_results
    ).execute()

    return [
        {
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        for item in response.get("items", [])
    ]

# =========================
# WIKIPEDIA
# =========================
wiki = wikipediaapi.Wikipedia(language="en", user_agent="UniSearchAI/1.0")

@st.cache_data(show_spinner=False)
def fetch_wiki(query):
    page = wiki.page(query)
    if not page.exists():
        return []

    summary = ". ".join(page.summary.split(". ")[:5])
    return [{
        "title": page.title,
        "description": summary,
        "link": page.fullurl
    }]

# =========================
# MOVIES
# =========================
@st.cache_data(show_spinner=False)
def fetch_movies(query, max_results=5):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    data = requests.get(url, params=params).json()

    return [
        {
            "title": item.get("title", ""),
            "description": item.get("overview", ""),
            "link": f"https://www.themoviedb.org/movie/{item['id']}"
        }
        for item in data.get("results", [])[:max_results]
    ]

# =========================
# EMBEDDINGS
# =========================
def create_embeddings(items):
    texts, clean_items = [], []

    for item in items:
        text = f"{item['title']} {item['description']}".strip()
        if text:
            texts.append(text)
            clean_items.append(item)

    if not texts:
        return None, []

    return model.encode(texts), clean_items

# =========================
# FAISS SEARCH
# =========================
def search_top_k(query, items, embeddings, k=5):
    query_vec = model.encode([query])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    _, idx = index.search(query_vec, k)
    return [items[i] for i in idx[0]]

# =========================
# AGGREGATOR
# =========================
def fetch_all_sources(query, filters):
    sources = {}
    if filters["books"]:
        sources["Books"] = fetch_books(query)
    if filters["youtube"]:
        sources["YouTube"] = fetch_youtube(query)
    if filters["theory"]:
        sources["Theory"] = fetch_wiki(query)
    if filters["movies"]:
        sources["Movies"] = fetch_movies(query)
    return sources


def search_all_sources(query, filters):
    final = {}
    sources = fetch_all_sources(query, filters)

    for name, items in sources.items():
        embeddings, clean_items = create_embeddings(items)
        if embeddings is None:
            final[name] = []
            continue

        final[name] = search_top_k(
            query, clean_items, embeddings, k=min(5, len(clean_items))
        )
    return final

# =========================
# LLM SUMMARY
# =========================
from openai import OpenAI, RateLimitError

def summarize_results(query, results):
    try:
        text = f"User searched for: {query}\n\n"
        for source, items in results.items():
            text += f"{source}:\n"
            for item in items[:3]:
                text += f"- {item['title']}\n"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a learning assistant."},
                {"role": "user", "content": f"Summarize this into guidance:\n{text}"}
            ]
        )
        return response.choices[0].message.content

    except RateLimitError:
        # 🔥 Fallback summary (NO LLM)
        summary = f"Here is a structured learning path for **{query}**:\n\n"
        for source, items in results.items():
            if items:
                summary += f"🔹 **{source}**: {items[0]['title']}\n"
        summary += "\n(LLM summary unavailable due to API quota)"
        return summary


# =========================
# STREAMLIT UI
# =========================
st.title("📚 UniSearch AI – One Search for Everything")
st.write("Search books, tutorials, theory, movies & videos using AI")

query = st.text_input("Enter your search query")

st.subheader("🔍 Select Sources")
c1, c2 = st.columns(2)
with c1:
    use_books = st.checkbox("📚 Books", True)
    use_youtube = st.checkbox("📺 YouTube", True)
with c2:
    use_theory = st.checkbox("📖 Theory", True)
    use_movies = st.checkbox("🎬 Movies", True)

filters = {
    "books": use_books,
    "youtube": use_youtube,
    "theory": use_theory,
    "movies": use_movies
}

if st.button("Search") and query.strip():
    st.success(f"Searching for: {query}")
    results = search_all_sources(query, filters)

    tabs = st.tabs(["📚 Books", "📺 YouTube", "📖 Theory", "🎬 Movies"])
    tab_keys = ["Books", "YouTube", "Theory", "Movies"]

    for tab, key in zip(tabs, tab_keys):
        with tab:
            items = results.get(key, [])
            if not items:
                st.info("No results found")
                continue

            for i, item in enumerate(items, 1):
                st.markdown(f"### {i}. {item['title']}")
                st.write(item["description"][:250] or "No description")
                st.markdown(f"[🔗 Open Link]({item['link']})")
                st.divider()

    with st.spinner("🤖 Generating AI summary..."):
        summary = summarize_results(query, results)

    st.subheader("🧠 AI Learning Summary")
    st.success(summary)
