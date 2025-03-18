from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.llms.ollama import Ollama
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from utils import get_embedding_function
from dotenv import load_dotenv
import os

os.environ["PINECONE_API_KEY"] = "c563e341-f430-41a5-8dc4-93596352b778"
os.environ["PINECONE_ENVIRONMENT_REGION"] = "us-east-1"
INDEX_NAME = "man"

metadata_field_info = [
    AttributeInfo(
        name="categoría",
        description="The category of the content, indicating the broader field or domain of the historical period or topic. Valid values include ['Historia', 'Arqueología', 'Arte']",
        type="string"
    ),
    AttributeInfo(
        name="etapa",
        description="The specific historical stage or period the content pertains to. For instance, 'Grecia' refers to the Greek civilization and its cultural context.",
        type="string"
    ),
    AttributeInfo(
        name="obra",
        description="The title or name of the work or piece being described, such as an introduction or a specific artifact.",
        type="string"
    ),
    AttributeInfo(
        name="procedencia",
        description="The origin or provenance of the work, indicating where it comes from. For example, 'Grecia' indicates it is from Greece.",
        type="string"
    ),
    AttributeInfo(
        name="recorrido",
        description="The thematic path or tour in which the work is included within the museum. This could indicate an introduction or a special route.",
        type="string"
    ),
    AttributeInfo(
        name="sala",
        description="The room in the museum where the piece is displayed. Be very precise when a query asks about the location or room, searching for the metadata obra and answering with this.",
        type="string"
    ),
    AttributeInfo(
        name="siglo",
        description="The time period or centuries that the work represents, such as 'Siglos XV a.C. - II a.C.', indicating it spans from the 15th century BC to the 2nd century BC.",
        type="string"
    ),
    AttributeInfo(
        name="tema",
        description="The specific theme or subject matter of the work, which provides additional context. For example, 'Grecia' refers to content focused on Greek history or culture.",
        type="string"
    )
]

# Adjusted the description for more precise retrieval of rooms and works
document_content_description = """
Overview of an artifact or exhibit from the Museo Arqueologico Nacional. If the query pertains to a room or location, 
ensure the search returns specific room details with precise room names, such as 'Sala 36'. 
If the query relates to a work of art or artifact, return the correct title and description. 
For thematic paths, include only relevant paths, such as 'Historia' or 'Arqueología', but avoid generalizations.
Limit the results to 5 items, prioritizing relevance.
Use the original query for processing.
"""

def get_fine_retriever():
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=get_embedding_function()
    )
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, enable_limit=True,
        search_kwargs={"k": 5}  # Reduced k to 5 for more focused results
    )
    return retriever




def get_general_retriever():
    embedding_function = get_embedding_function()
    db = PineconeVectorStore(embedding=embedding_function, index_name=INDEX_NAME)
    return db
