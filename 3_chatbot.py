import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.
Answer the user's question using only the retrieved context below.
If the answer is not in the context, say "I don't know".

For every factual claim, include the source URL.

Retrieved context:
{context}"""
    ),
    ("human", "{input}")
])

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_question = st.chat_input("Ask a question")

if user_question:
    st.session_state.messages.append(HumanMessage(content=user_question))

    with st.chat_message("user"):
        st.markdown(user_question)

    retrieved_docs = vector_store.similarity_search(user_question, k=2)

    context_parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "unknown")
        context_parts.append(
            f"Title: {title}\nSource: {source}\nContent: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    chain = prompt | llm
    response = chain.invoke({
        "input": user_question,
        "context": context,
    })

    ai_message = response.content

    st.session_state.messages.append(AIMessage(content=ai_message))

    with st.chat_message("assistant"):
        st.markdown(ai_message)