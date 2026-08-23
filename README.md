# 📺 YouTube AI RAG Chatbot

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)

A powerful, interactive web application that allows you to chat with any YouTube video. Just paste a URL, and the AI will extract the transcript, index the content, and answer any questions you have with strict factual accuracy using a **Retrieval-Augmented Generation (RAG)** architecture.

---

## ✨ Features
- **Multilingual Support**: Supports both English and Hindi YouTube videos natively.
- **Fast Vector Search**: Uses local `FAISS` indexing for lightning-fast retrieval of context.
- **Hallucination Prevention**: Configured with `temperature=0` to ensure the AI only answers based on the video's actual transcript.
- **Modern UI**: Built with Streamlit for a clean, responsive, ChatGPT-like chat interface.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM Engine**: Google Gemini (gemini-3.5-flash)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Orchestration**: LangChain (LCEL)

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/mahaveer0738/YouTube_Chatbot.git
cd YouTube_Chatbot
```

### 2. Set up the virtual environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GOOGLE_API_KEY="AIzaSyYourSecretKeyHere..."
```

### 5. Launch the App
```bash
streamlit run streamlit_app.py
```

---

## ☁️ Deployment (Streamlit Cloud)
1. Fork or push this repository to your GitHub account.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app pointing to `streamlit_app.py`.
4. In the "Advanced Settings", add your Google API key to the Secrets panel:
   ```toml
   GOOGLE_API_KEY="AIza..."
   ```
5. Click **Deploy!**

---
*Built with ❤️ using LangChain and Streamlit.*
