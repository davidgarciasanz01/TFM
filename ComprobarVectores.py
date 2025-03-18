import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configuración de Pinecone (usa las mismas credenciales que en el primer script)
os.environ["PINECONE_API_KEY"] = "c563e341-f430-41a5-8dc4-93596352b778"
os.environ["PINECONE_ENVIRONMENT_REGION"] = "us-east-1"
index_name = "man"

# Define el prompt que deseas comparar
prompt = "recorrido imprescindibles."

# Genera el embedding del prompt
embedding_model = OllamaEmbeddings(model='nomic-embed-text')
prompt_embedding = embedding_model.embed_query(prompt)

# Conéctate a la base de datos vectorial en Pinecone
# Conéctate a la base de datos vectorial en Pinecone
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model  # Proporciona aquí el modelo de embeddings
)

# Realiza una búsqueda en Pinecone utilizando el embedding generado
# Realiza una búsqueda de similitud usando el texto del prompt directamente
results = vectorstore.similarity_search(query=prompt, k=5)
# `k=5` es la cantidad de resultados similares que deseas obtener.

# Extrae el contenido de los documentos similares
similar_docs = [doc.page_content for doc in results]

# Combina el contenido de los documentos relevantes en un solo texto
combined_content = "\n".join(similar_docs)

print(f"Los documentos más similares al prompt '{prompt}' son:")
print(combined_content)

# Si deseas usar un modelo generativo para refinar la respuesta (opcional):
# from langchain_core.chains import ConversationalChain
# generative_model = ConversationalChain(model='tu_modelo_generativo')  # Reemplaza con tu modelo configurado
# response = generative_model.generate_response(input_text=prompt, context=combined_content)
# print(f"Respuesta basada en la búsqueda RAG:\n{response}")
