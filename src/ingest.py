import chromadb
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

LITERATURE_DIR = Path("data/literature")
CHROMA_DIR = Path("data/chroma_db")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrae texto de un PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def ingest_papers():
    # Setup ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Borrar colección si existe (para re-ingest limpio)
    try:
        client.delete_collection("clinical_literature")
    except:
        pass
    
    collection = client.create_collection("clinical_literature")
    
    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    pdfs = list(LITERATURE_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} papers\n")
    
    all_chunks = []
    all_ids = []
    all_metadata = []
    
    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")
        try:
            text = extract_text_from_pdf(pdf_path)
            chunks = splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_path.stem}_{i}")
                all_metadata.append({"source": pdf_path.name})
            
            print(f"  {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Ingest en batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size]
        )
    
    print(f"\nTotal chunks ingested: {len(all_chunks)}")
    print(f"ChromaDB saved to {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_papers()