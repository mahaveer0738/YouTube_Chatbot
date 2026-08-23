import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

def extract_video_id(url: str) -> str:
    """Extracts the YouTube video ID from a given URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return url

def index_youtube_video(video_url: str, save_path: str = "faiss_index"):
    """
    Fetches transcript from a YouTube video, chunks it, generates embeddings,
    and stores them in a FAISS vector store.
    """
    print(f"Loading transcript for: {video_url}")
    try:
        # We are using LangChain's YoutubeLoader, which uses youtube-transcript-api under the hood.
        # It automatically extracts the video ID and converts the transcript into LangChain Documents.
        # You can specify languages just like in the video:
        loader = YoutubeLoader.from_youtube_url(
            video_url, 
            add_video_info=False,
            language=["en", "en-US"] # Specify preferred languages here
        )
        documents = loader.load()
    except Exception as e:
        print(f"Error loading video: {e}")
        return None

    if not documents:
        print("No transcript found.")
        return None
    
    print("Transcript loaded successfully.")
    
    # --- TEMPORARY PRINT FOR YOU TO SEE THE TRANSCRIPT ---
    # print("\n" + "="*50)
    # print("TRANSCRIPT PREVIEW (First 1500 chars):")
    # print(documents[0].page_content[:1500])
    # print("="*50 + "\n")
    # -----------------------------------------------------

    # Step 2: Split the text into manageable chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # Step 3 & 4: Generate Embeddings and Store in FAISS
    # Note: Requires GOOGLE_API_KEY to be set in environment variables
    print("Generating embeddings and building vector store (this might take a moment)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save the vector store locally for later retrieval
    vector_store.save_local(save_path)
    print(f"Vector store successfully saved locally to '{save_path}' directory.")
    
    return vector_store

if __name__ == "__main__":
    # Test the indexing part (you need to provide your GOOGLE_API_KEY in .env first)
    # Example video: a short explanatory video
    test_url = "https://www.youtube.com/watch?v=crH7kpjomIk" 
    index_youtube_video(test_url)
