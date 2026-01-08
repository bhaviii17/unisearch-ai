# 🔍 UniSearch AI  
### One Search. Every Learning Resource.

📚 **UniSearch AI** is an AI-powered unified search and recommendation platform that helps users discover the **most relevant books, videos, articles, and movies** from a single natural-language query.

🔗 **Live Demo:** https://unisearch-ai-mcc3ez6vcyq2s78emdju3f.streamlit.app/

---

## 🚀 What Can UniSearch AI Do?

Ask questions like:
- *“Best resources to learn data science”*
- *“Movies explaining artificial intelligence”*
- *“Beginner-friendly Python tutorials”*

UniSearch AI intelligently retrieves and ranks content across multiple platforms — saving hours of manual searching.

---

## 🧠 How It Works (Under the Hood)

This project follows a **modern Retrieval + Ranking architecture**:

🔹 **Bi-Encoder (Sentence Transformers)**  
→ Converts queries & content into vector embeddings  

🔹 **FAISS Vector Search**  
→ Fast semantic retrieval of relevant results  

🔹 **Multi-Source Aggregation**  
→ Books, YouTube, Wikipedia, Movies APIs  

🔹 **Optional LLM Summarization**  
→ Generates concise learning insights *(graceful fallback if quota exhausted)*  

🔹 **Streamlit UI**  
→ Interactive, fast, and deployable web interface  

---

## ✨ Key Features

✅ Natural language search  
✅ Semantic (meaning-based) ranking  
✅ Multi-source recommendations  
✅ Tab-based UI with filters  
✅ Fast performance with caching  
✅ Production-ready deployment setup  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Sentence Transformers**
- **FAISS**
- **OpenAI API**
- **Google Books API**
- **YouTube Data API**
- **TMDB API**
- **Git & GitHub**

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/USERNAME/unisearch-ai.git
cd unisearch-ai
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
