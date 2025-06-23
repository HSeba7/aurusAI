import openai
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings  # Still uses langchain_openai
from langchain_community.vectorstores import Pinecone

from langchain_community.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os

def read_doc(directory):
    # Check if directory exists and contains PDFs
    if not os.path.isdir(directory):
        print(f"The path '{directory}' is not a directory.")
        return []
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in directory '{directory}'.")
        return []
    
    # Load PDFs
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    # print("Loaded documents:", documents)
    return documents

doc=read_doc(r'C:\Users\RTX\Desktop\python2023\sebastian\a')

def chunk_data(docs,chunk_size=1500,chunk_overlap=300):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc

documents=chunk_data(docs=doc)

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

vectors=embeddings.embed_query("How are you?")


from pinecone import ServerlessSpec

pc = pinecone.Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    environment="us-east-1-aws"  # Replace with your specific environment
)

index_name = "chatbot"

# Check if the index exists and create if necessary
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Make sure this matches your embedding dimension
        metric="cosine",  # Adjust to your desired similarity metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the Pinecone index using Langchain's wrapper
index = Pinecone.from_documents(documents, embeddings, index_name=index_name)

# index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

def retrieve_query(query,k=1):
    matching_results=index.similarity_search(query,k=k)
    return matching_results

from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
llm=ChatOpenAI(model_name="gpt-4o")
chain=load_qa_chain(llm,chain_type="stuff")

## Search answers from VectorDB
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    input_data = {
        "input_documents": doc_search,
        "question": query
    }
    response=chain.invoke(input=input_data)
    answer = response.get("output_text", "No answer found.")
    return answer.strip()


our_query = "Jakie korzyści przynosi pozytywna kultura organizacji?"
answer = retrieve_answers(our_query)
print("================",answer)