from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_pinecone import PineconeEmbeddings
import re
import asyncio

def get_embedding_function():
    # Inicializar embeddings con Pinecone
    embeddings = PineconeEmbeddings(
        model="multilingual-e5-large",  # Modelo de embeddings
        pinecone_api_key="c563e341-f430-41a5-8dc4-93596352b778",  # Reemplaza con tu API key de Pinecone
        index_name="man2"          # Nombre del índice que creaste en Pinecone
    )
    return embeddings
