import pinecone
import os
from langchain_community.embeddings import OllamaEmbeddings

os.environ["PINECONE_API_KEY"] = "c563e341-f430-41a5-8dc4-93596352b778"
os.environ["PINECONE_ENVIRONMENT_REGION"] = "us-east-1"
index_name = "man"
# Conéctate al índice de Pinecone usando la API nativa
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT_REGION"))
index = pinecone.Index(index_name)

prompt = "Describe la historia de la Edad Moderna."



embedding_model = OllamaEmbeddings(model='nomic-embed-text')
prompt_embedding = embedding_model.embed_query(prompt)

# Realiza la búsqueda de similitud con el embedding
response = index.query(
    vector=prompt_embedding,
    top_k=5,  # Número de resultados que quieres
    include_metadata=True  # Incluye los metadatos en los resultados
)

# Procesa los resultados para extraer el contenido relevante
similar_docs = [match['metadata']['text'] for match in response['matches']]

# Combina el contenido de los documentos relevantes en un solo texto
combined_content = "\n".join(similar_docs)

print(f"Los documentos más similares al prompt '{prompt}' son:")
print(combined_content)
