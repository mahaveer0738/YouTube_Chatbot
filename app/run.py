import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.retrieval import get_retriever
from app.augmentation import get_prompt_template

# Load environment variables
load_dotenv()

from langchain_core.runnables import RunnablePassthrough

# Helper function to format the documents into a single string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_chatbot():
    print("==================================================")
    print("Welcome to the YouTube RAG Chatbot!")
    print("Type 'exit' or 'quit' to stop.")
    print("==================================================")

    # 1. Initialize the components
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    retriever = get_retriever()
    prompt_template = get_prompt_template()
    
    if not retriever:
        print("Failed to load retriever. Exiting.")
        return

    # 2. Build the Sequential Runnable Chain (LCEL)
    # The '|' symbol acts as a pipe, passing data from left to right!
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    # Interactive Loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue

        try:
            # Now we only need to call invoke ONCE on our chain!
            answer = rag_chain.invoke(user_input)
            
            # Since the LLM is the last step in the chain, it returns an AIMessage object
            print("\nBot Answer Content:")
            
            # The new Gemini SDK returns a list containing the text AND a signature (watermark)
            # We will extract just the text so you don't see the ugly 'extras' block!
            if isinstance(answer.content, list) and len(answer.content) > 0:
                clean_text = answer.content[0].get('text', '')
                print(clean_text)
            else:
                print(answer.content)
            
            print("\n--- Answer Metadata ---")
            print(answer.response_metadata)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()
