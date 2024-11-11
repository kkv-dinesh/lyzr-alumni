from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lyzr import QABot
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please add it to the .env file.")

# Set the OpenAI API key for lyzr
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the QABot with the PDF
my_chatbot = QABot.pdf_qa(input_files=["alumni_details.pdf"])

# Define FastAPI app
app = FastAPI()

# Define request body schema
class QueryRequest(BaseModel):
    query: str

# Define the endpoint
@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Get the response from QABot
        response = my_chatbot.query(request.query)
        return {"query": request.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
