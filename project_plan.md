# YouTube Chatbot Web Interface Plan

We are building a beautiful, interactive web interface using **Streamlit** for our powerful RAG engine. 

### Why Streamlit?
- **Python Only:** Streamlit lets us build the entire website using only Python.
- **Fast & Beautiful:** Streamlit has built-in chat UI components (`st.chat_message`, `st.chat_input`) that make it incredibly easy to build a ChatGPT-like interface.
- **Deployment:** Streamlit apps are extremely easy to deploy to the web for free using **Streamlit Community Cloud**.

## File Structure

```text
app/
  __init__.py
  indexing.py
  retrieval.py
  augmentation.py
  run.py
.env
streamlit_app.py   <-- The new professional web interface
```

## Implementation Details

### Step 1: Install Dependencies
- Run `pip install streamlit` in the virtual environment.

### Step 2: Build `streamlit_app.py`
The web app will have two main areas:

**1. The Sidebar (Setup Area):**
- A professional header and description.
- A text input box where the user pastes the YouTube URL.
- A "Process Video" button.
- When clicked, it calls `index_youtube_video()` from `app/indexing.py` and shows a loading spinner until it's done.

**2. The Main Chat Interface:**
- We use Streamlit's session state (`st.session_state`) to remember the chat history.
- A chat input box at the bottom.
- When a user asks a question, we:
  - Load the FAISS index using `get_retriever()` from `app/retrieval.py`.
  - Pass the question through the manual LLM generation process.
  - Display the Bot's answer in the chat window with a nice avatar.

### Step 3: Deployment (Future Step)
1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and point it to `streamlit_app.py`.
4. Add your `GOOGLE_API_KEY` to the Streamlit Secrets panel.
5. Hit Deploy!
