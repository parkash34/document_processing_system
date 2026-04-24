# Bella Italia — Document Processing System

A document processing and RAG system built with FastAPI, LangChain
and Pinecone. Automatically loads and processes real documents —
PDF menus and FAQ text files — making them searchable by meaning.
Restaurant owners can update documents without changing any code.

## Features

- PDF and TXT document loading — real files not hardcoded data
- Automatic embedding pipeline — load, split, embed and store in one flow
- Pinecone vector database — embeddings stored permanently in cloud
- Source tracking — every answer shows which document it came from
- Cross document search — searches across all documents simultaneously
- Live update endpoint — refresh AI knowledge without restarting server
- RAG answers — AI answers only from real document content
- Input validation — empty queries rejected automatically

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend web framework |
| LangChain | Document processing and RAG framework |
| Pinecone | Cloud vector database |
| HuggingFace | Free local embedding model |
| Groq API | AI language model |
| LLaMA 3.3 70B | AI model |
| PyPDF | PDF document reading |
| Pydantic | Data validation |
| python-dotenv | Environment variable management |

## Project Structure
```
document-processing/
│
├── env/
├── main.py
├── menu.pdf
├── faq.txt
├── .env
└── requirements.txt
```

## Setup

1. Clone the repository
```
git clone https://github.com/yourusername/bella-italia-document-processing
```
2. Create and activate virtual environment
```
python -m venv env
env\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Create `.env` file and add your API keys
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

5. Add your documents to project folder
```
menu.pdf  →  restaurant menu PDF
faq.txt   →  frequently asked questions text file
```

6. Run the server
```
uvicorn main:app --reload
```

## API Endpoints

### POST /ask
Answers customer questions using document content.

**Request:**
```json
{
    "query": "Do you have vegan options?"
}
```

**Response:**
```json
{
    "answer": "Yes, we have Vegan Arrabbiata — a spicy tomato pasta with no animal products, priced at $12."
}
```

### POST /sources
Shows which documents and pages were used to find the answer.

**Request:**
```json
{
    "query": "Do you have vegetarian pizza?"
}
```

**Response:**
```json
{
    "sources": [
        {
            "content": "Vegetarian Pizza - mixed vegetables...",
            "source": "menu.pdf",
            "page": 0
        }
    ]
}
```

### POST /update
Refreshes AI knowledge when documents are updated.
No request body needed.

**Response:**
```json
{
    "message": "Documents updated successfully!"
}
```

## How It Works
```
menu.pdf + faq.txt
↓
PyPDFLoader and TextLoader load files
↓
RecursiveCharacterTextSplitter splits into chunks
chunk_size=500, chunk_overlap=50
↓
HuggingFace converts chunks to 384 dimension embeddings
↓
Stored permanently in Pinecone cloud index
↓
User sends query
↓
Query converted to embedding
↓
Pinecone finds similar chunks across all documents
↓
Chunks sent to AI with source information
↓
AI generates accurate answer from real documents
```

## Updating Documents
```
When restaurant owner updates the menu:

Replace menu.pdf with new file
Call POST /update
System automatically reprocesses all documents
AI uses updated information immediately


No code changes needed — this is how the SaaS business works.
```
## Document Pipeline

```python
# Automatic pipeline
PyPDFLoader  →  loads PDF pages as Documents
TextLoader   →  loads TXT file as Document
↓
split_documents()  →  splits into chunks preserving metadata
↓
add_documents()    →  embeds and stores in Pinecone
```

## Metadata Tracking
```
Each chunk stored with source information:
PDF chunks:
source: "menu.pdf"
page: 0
TXT chunks:
source: "faq.txt"
page: N/A
```

## Environment Variables
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Notes

- Never commit your .env file to GitHub
- Documents embedded once — skipped on subsequent restarts
- Call /update endpoint after changing any document
- System searches across all documents simultaneously
- HuggingFace model downloads automatically on first run
- Pinecone free tier — 1 index, 2GB storage, no credit car

## 👤 Author

**Ohm Parkash** — [LinkedIn](https://www.linkedin.com/in/om-parkash34/) · [GitHub](https://github.com/parkash34)