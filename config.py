import os
from dotenv import load_dotenv

load_dotenv()

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-rag-sys") 