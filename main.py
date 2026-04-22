import os
from pydantic import BaseModel, field_validator
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API KEY is missing in .env file")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE API KEY is missing in .env file")

app = FastAPI()

class Query(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Query is Empty")
        return v

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=pinecone_api_key)

if "bella-italia-docs" not in pc.list_indexes().names():
    pc.create_index(
        name="bella-italia-docs",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

def build_restaurant_pipeline():
    all_documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    menu_loader = PyPDFLoader("menu.pdf")
    menu_docs = menu_loader.load()
    print(f"Menu pages loaded: {len(menu_docs)}")
    menu_chunks = splitter.split_documents(menu_docs)
    all_documents.extend(menu_chunks)


    faq_loader = TextLoader("faq.txt")
    faq_docs = faq_loader.load()
    print(f"FAQ documents loaded: {len(faq_docs)}")
    faq_chunks = splitter.split_documents(faq_docs)

    all_documents.extend(faq_chunks)
    print(f"Total chunks: {len(all_documents)}")

    return all_documents


vector_store = PineconeVectorStore(
    index_name="bella-italia-docs",
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)

index = pc.Index("bella-italia-docs")
stats = index.describe_index_stats()

if stats.total_vector_count == 0:
    chunks = build_restaurant_pipeline()
    vector_store.add_documents(chunks)
    print("Documents loaded successfully")
else:
    print("Documents already loaded - skipping")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.2,
    max_tokens = 500,
    api_key = api_key
)

@app.post("/ask")
def ask_ai(query: Query):
    query = query.query

    results = vector_store.similarity_search(query, k=4)
    context = ""
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context += f"[Source: {source}]\n{doc.page_content}\n\n"

    prompt = f"""You are a restaurant assistant for Bella Italia.
    Answer the customer question using ONLY the information below.
    If the answer is not in the information say:
    "I don't have that information. Please call us at 123-456-7890"

    Information:
    {context}

    Customer Question: {query}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}

@app.post("/sources")
def sources_ai(query: Query):
    query = query.query
    results = vector_store.similarity_search(query, k=4)
    formatted = []
    for doc in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A")
        })
    return {"sources": formatted}

@app.post("/update")
def update_ai():
    try:
        pc.delete_index("bella-italia-docs")
        print("Old index deleted")

        pc.create_index(
            name="bella-italia-docs",
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print("New index created")

        global vector_store
        vector_store = PineconeVectorStore(
            index_name="bella-italia-docs",
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key
        )

        chunks = build_restaurant_pipeline()
        vector_store.add_documents(chunks)

        return {"message": "Documents updated successfully!"}
    except Exception as e:
        return {"error": f"Update failed {str(e)}"}

    