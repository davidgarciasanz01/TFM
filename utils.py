from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_pinecone import PineconeEmbeddings
import re



from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://tuservidor:11434")
    return embeddings


