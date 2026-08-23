import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_retriever(save_path: str = "faiss_index"):
    """
    Loads the saved FAISS index from disk and returns it as a LangChain retriever.
    """
    print(f"Loading vector store from '{save_path}'...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        # Configure the retriever (e.g., fetch top 4 most relevant chunks)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("Retriever successfully loaded.")
        return retriever
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please ensure you have run the indexing step first.")
        return None

if __name__ == "__main__":
    # Test the retrieval part
    retriever = get_retriever()
    if retriever:
        query = "When the world war 2 started and when it ended?"
        docs = retriever.invoke(query)
        print(f"\nFound {len(docs)} relevant documents for the query: '{query}'")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content[:200] + "...")
