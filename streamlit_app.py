import streamlit as st
import os
from dotenv import load_dotenv

from app.indexing import index_youtube_video
from app.retrieval import get_retriever
from app.augmentation import get_prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube AI Assistant", 
    page_icon="📺", 
    layout="wide"
)

# Custom CSS to make it look premium
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF0000;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555555;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    /* Hide the top right default menu for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize session state variables to remember chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_indexed" not in st.session_state:
    # If a vector store already exists on disk from previous runs, we can assume it's indexed
    if os.path.exists("faiss_index"):
        st.session_state.is_indexed = True
    else:
        st.session_state.is_indexed = False

# ---------------------------------------------------------
# Sidebar (Configuration Area)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=150)
    st.markdown("### ⚙️ Video Setup")
    
    youtube_url = st.text_input("Paste YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Process Video", use_container_width=True, type="primary"):
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Extracting transcript and analyzing video... This may take a minute."):
                # Call our backend indexing engine
                vector_store = index_youtube_video(youtube_url)
                if vector_store:
                    st.session_state.is_indexed = True
                    # Clear old chat history when a new video is processed
                    st.session_state.messages = []
                    st.success("Video processed successfully! You can now chat.")
                else:
                    st.error("Failed to process video. Please check the URL.")
                    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This application uses a Retrieval-Augmented Generation (RAG) architecture built with LangChain and Google Gemini. It allows you to chat naturally with the transcript of any YouTube video.")

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.markdown('<p class="main-header">YouTube AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about the video you just processed!</p>', unsafe_allow_html=True)

# 1. Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input box
if prompt := st.chat_input("Ask a question about the video..."):
    
    # Ensure the user has indexed a video before allowing them to chat
    if not st.session_state.is_indexed:
        st.warning("Please process a YouTube video in the sidebar first!")
        st.stop()

    # Display user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Load Retriever and Prompt Template
                retriever = get_retriever()
                prompt_template = get_prompt_template()
                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
                
                if not retriever:
                    st.error("Could not load the database. Please try processing the video again.")
                    st.stop()

                # Step 1: Retrieve relevant documents (chunks)
                docs = retriever.invoke(prompt)
                
                # Step 2: Combine the retrieved documents into a single text string
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Step 3: Format the final prompt
                final_prompt = prompt_template.invoke({"context": context, "input": prompt})
                
                # Step 4: Get response from Gemini
                answer = llm.invoke(final_prompt)
                
                # Step 5: Clean up content if it's in the weird list format (Google SDK bug workaround)
                if isinstance(answer.content, list) and len(answer.content) > 0:
                    clean_text = answer.content[0].get('text', '')
                else:
                    clean_text = answer.content
                    
                # Display the response on the UI
                st.markdown(clean_text)
                
                # Save the response to session state history
                st.session_state.messages.append({"role": "assistant", "content": clean_text})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
