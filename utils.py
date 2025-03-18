#from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_pinecone import PineconeEmbeddings
import re
from langchain_ollama import OllamaEmbeddings


import asyncio


def get_embedding_function():
    # Asegurar que hay un bucle de eventos activo
    #try:
   #     asyncio.get_running_loop()
   # except RuntimeError:
    #    asyncio.set_event_loop(asyncio.new_event_loop())

    # Inicializar embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    #PineconeEmbeddings(model="multilingual-e5-large")
    return embeddings


