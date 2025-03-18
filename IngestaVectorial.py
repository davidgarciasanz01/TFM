import re
import os
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeEmbeddings

os.environ["PINECONE_API_KEY"] = "c563e341-f430-41a5-8dc4-93596352b778"
os.environ["PINECONE_ENVIRONMENT_REGION"] = "us-east-1"
index_name = "man2"

# Define la ruta de la carpeta donde están los archivos
base_directory = 'C:/Users/david/Desktop/MSC DATA SCIENCE/2 CUATRIMESTRE/TFM/MAN/Etapas'

def extract_metadata_and_description(text):
    """
    Extracts metadata fields and description from the text.
    """
    metadata_patterns = {
    'Recorrido': r'Recorrido:\s*([^\n]*)',
    'Obra': r'Obra:\s*([^\n]*)',
    'Sala': r'Sala:\s*([^\n]*)',
    'Etapa': r'Etapa:\s*([^\n]*)',
    'Siglo': r'Siglo:\s*([^\n]*)',
    'Categoría': r'Categoría:\s*([^\n]*)',
    'Procedencia': r'Procedencia:\s*([^\n]*)',
    'Tema': r'Tema:\s*([^\n]*)'
}

    
    metadata = {}
    # Extract each metadata field using regex
    for key, pattern in metadata_patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = match.group(1).strip() or ""  # Use empty string if match is empty
        else:
            metadata[key] = ""  # Use empty string if no match

    # Extract description by removing metadata lines from the text
    description = re.sub('|'.join(metadata_patterns.values()), '', text).strip()
    metadata['text'] = description

    return metadata

def generate_document(file_path):
    """
    Reads a text file, extracts metadata, and creates a Document object.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    metadata = extract_metadata_and_description(text)
    doc = Document(page_content=metadata['text'], metadata=metadata)
    return doc

def process_all_files_in_directory(directory):
    """
    Recursively processes all text files in the given directory.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                doc = generate_document(file_path)
                documents.append(doc)
    return documents

def split_documents(input_documents):
    """
    Splits the data into chunks divided by new lines.
    """
    print("SPLITTING DATA: Starting...")
    text_splitter = CharacterTextSplitter(
    chunk_size=800,  # Tamaño máximo de cada chunk en caracteres
    chunk_overlap=100,  # Superposición de 200 caracteres entre chunks
    length_function=len,
    is_separator_regex=False,  # Mantén esto en False si no usas un separador
)

    chunks = text_splitter.split_documents(input_documents)
    print(f"SPLITTING DATA: Finished with {len(chunks)} chunks.")
    return chunks

def insert_documents_to_pinecone(index_name, docs):
    """
    Inserts the data into the Pinecone index.
    """
    print("INSERTING INTO PINECONE: Starting...")
    vectorstore = PineconeVectorStore.from_documents(
        docs, PineconeEmbeddings(model="multilingual-e5-large"), index_name=index_name
    )
    print("INSERTING INTO PINECONE: Finished.")

if __name__ == '__main__':
    # Process all files and generate documents
    documents = process_all_files_in_directory(base_directory)
    # Split the documents into chunks
    documents = split_documents(documents)
    # Insert the documents into Pinecone
    insert_documents_to_pinecone(index_name, documents)
