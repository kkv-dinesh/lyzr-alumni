from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack.nodes import FARMReader, PDFToTextConverter
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
import os
from glob import glob

# FastAPI app setup
app = FastAPI()

# Define the request body schema
class QueryRequest(BaseModel):
    query: str
    folder_path: str  # Path to the folder containing PDFs

# Initialize Haystack components
document_store = InMemoryDocumentStore()

# Function to load PDFs into the document store
def load_pdfs_to_store(folder_path: str):
    pdf_files = glob(os.path.join(folder_path, "*.pdf"))
    pdf_converter = PDFToTextConverter()
    
    for pdf_file in pdf_files:
        docs = pdf_converter.convert(file_path=pdf_file)
        document_store.write_documents(docs)

# Initialize the QA model (use a pre-trained model like `deepset/roberta-base-squad2`)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Create a QA pipeline
qa_pipeline = ExtractiveQAPipeline(reader, document_store)

# Query handler
@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Load all PDFs in the folder into the document store
        load_pdfs_to_store(request.folder_path)
        
        # Perform the query using the Haystack pipeline
        result = qa_pipeline.run(query=request.query, top_k_retriever=3, top_k_reader=3)
        
        # Extract and return the best answer
        answer = result["answers"][0].answer if result["answers"] else "No answer found."
        return {"query": request.query, "response": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
