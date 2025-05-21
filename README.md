# **HR Chatbot with RAG**  
An intelligent HR chatbot that conducts initial screening interviews using Retrieval-Augmented Generation (RAG) to personalize conversations based on candidate resumes stored in a vector database.

## **Prerequisites** 
Python 3.9+\n
Pinecone API key
Groq API key
.env file with your credentials

## **Installation**
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Create a .env file:
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=hr-rag-sys
   PINECONE_ENVIRONMENT=your_pinecone_environment


## **To Run the Application:**
1. Data Ingestion:
python ingest.py

2. Running the Interview:
python main.py


