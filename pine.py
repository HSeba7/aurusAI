import os
from pinecone import Pinecone, ServerlessSpec
import pinecone
import openai
import pdfplumber

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECODE")
# Initialize Pinecone instance with the API key and environment details
pc = Pinecone(api_key= pinecone_api_key)

# Specify serverless specifications (e.g., for AWS in us-east-1 region)
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Check if the index exists, and create if necessary
if 'chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='chatbot',
        dimension=1536,
        metric='cosine',
        spec=spec
    )

# Connect to the existing index
index = pc.Index('chatbot')

# Now you can perform operations on `index`


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to generate embedding
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Load PDF, generate embedding, and upload to Pinecone
pdf_text = extract_text_from_pdf(r"C:\Users\RTX\Desktop\python2023\sebastian\a")
embedding = generate_embedding(pdf_text)
doc_id = "example_document"
metadata = {"filename": "example.pdf", "description": "Sample document for testing"}
index.upsert([(doc_id, embedding, metadata)])
print(f"Document '{doc_id}' uploaded to Pinecone.")
