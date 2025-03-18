from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from Retriever import get_fine_retriever
import os

import streamlit as st

# Acceder a las claves almacenadas en secrets.toml
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENVIRONMENT_REGION = st.secrets["pinecone"]["environment_region"]
#GROQ_API_KEY = st.secrets["groq"]["api_key"]

# Configuración del LLM
INDEX_NAME = "man"
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key="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")
)

# Prompt del sistema con instrucciones claras
_system_guide_prompt = """
Always answer in spanish
You are a virtual assistant for the National Archaeological Museum of Madrid.
Always use the conversation history to respond.
When the user ask for a room you have the value of the metadata Sala.
Start by indicating to the user what you do.
The chat history includes details about the current conversation; rely on it to maintain a smooth interaction.
Respond clearly, briefly, and accurately based on information about the museum.
You can only answer using the context provided, and you must be very precise with the information you share.
If you think the user has mistaken a name, be polite and ask if they are referring to the correct artwork.
Assist with questions related to artworks, rooms, tours, historical periods, and other topics.
It is very important that you do not make up answers; if unsure, ask for clarification about the artwork in question.
However, when mentioning an artwork, ensure it is one located in this museum. Variations in capitalization should not matter.
When asked about a tour, verify if it corresponds to the mentioned ones; if not, suggest one from the available list.

{context}
"""

# Prompt para contextualizar preguntas
_contextualize_q_system_prompt = """
Use the chat history to answer questions.
Rely on previous questions to maintain a fluid and engaging conversation, ensuring it is not tedious for the person interacting with you. If the user asks about a specific topic, continue discussing that topic until they mention a different one.
Retain information from previous questions to ensure continuity and context in the discussion.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def run_guide_llm(query: str, chat_history: ChatMessageHistory):
    retriever = get_fine_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _system_guide_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    qa_chain = qa_chain.pick('answer')

    response = qa_chain.invoke({"input": query, "chat_history": chat_history})
    return response

    

