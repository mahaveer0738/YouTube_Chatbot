from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    """
    Returns the prompt template for the RAG chatbot.
    """
    system_prompt = (
        "You are an assistant for question-answering tasks based on YouTube video transcripts. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    return prompt
