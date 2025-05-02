import os
import json
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Local file path for JSON
JSON_PATH = "dummy-resume.json"

def generate_resume_text(data):
    parts = [
        f"Name: {data.get('name', '')}",
        f"Phone: {data.get('phone', '')}",
        f"Email: {data.get('email', '')}",
        f"Location: {data.get('location', '')}",
        f"Experience: {data.get('experience_years', 0)} years",
        f"Skills: {', '.join(data.get('skills', []))}",
        f"Current Role: {data.get('current_role', '')} at {data.get('company', '')}",
        f"Education: {data.get('education', '')}",
    ]
    for project in data.get("projects", []):
        parts.append(f"Project: {project['title']} - {project['description']}")
    return "\n".join(parts)

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # matches all-MiniLM-L6-v2 model
            metric="cosine"
        )
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        # Wait for index to be ready
        time.sleep(60)
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
    
    return pc.Index(PINECONE_INDEX_NAME)

def process_candidate(model, pinecone_index, candidate_id, candidate_data):
    name = candidate_data.get("name", "")
    phone_number = candidate_data.get("phone", "")
    resume_text = candidate_data.get("resume_text", "")

    if not name or not phone_number:
        print(f"Missing required data for candidate {candidate_id}")
        return False

    if not resume_text:
        resume_text = generate_resume_text(candidate_data)

    chunk_size = 1000
    chunks = [resume_text[i:i+chunk_size] for i in range(0, len(resume_text), chunk_size)]
    if not chunks:
        chunks = [resume_text]

    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).astype("float32").tolist()
        vector_id = f"{candidate_id}_chunk_{i}"
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "candidate_id": candidate_id,
                "name": name,
                "phone": phone_number,
                "chunk_id": str(i),  # Convert to string
                "text": chunk,
                "is_phone_entry": "false"  # Convert to string
            }
        })

    # Add phone number as separate searchable item
    phone_query = f"Phone number: {phone_number}"
    phone_embedding = model.encode(phone_query).astype("float32").tolist()
    phone_vector_id = f"{candidate_id}_phone"
    
    vectors.append({
        "id": phone_vector_id,
        "values": phone_embedding,
        "metadata": {
            "candidate_id": candidate_id,
            "name": name,
            "phone": phone_number,
            "chunk_id": "-1",  # Special value for phone entries
            "text": phone_query,
            "is_phone_entry": "true"  # Convert to string
        }
    })

    # Upsert vectors in batches (Pinecone recommends batches of 100 or less)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting batch {i//batch_size}: {e}")
            return False

    return True

def ingest_candidates(json_path):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Initializing Pinecone...")
    pinecone_index = initialize_pinecone()

    with open(json_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    if isinstance(candidates, dict):  
        candidates = list(candidates.values())

    print(f"Found {len(candidates)} candidates")

    success_count = 0
    for i, candidate_data in tqdm(enumerate(candidates), desc="Processing candidates"):
        candidate_id = f"candidate_{i+1}"
        if process_candidate(model, pinecone_index, candidate_id, candidate_data):
            success_count += 1

    print(f"Successfully processed {success_count} out of {len(candidates)} candidates")
    print("Indexing complete. You can now query the Pinecone index.")

    return model, pinecone_index

def test_phone_lookup(model, pinecone_index, phone_number):
    query = f"Phone number: {phone_number}"
    query_vector = model.encode(query).astype("float32").tolist()

    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )

        if results.matches:
            match = results.matches[0]
            meta = match.metadata
            if meta.get("is_phone_entry"):
                print(f"\nFound candidate:")
                print(f"Name: {meta['name']}")
                print(f"Phone: {meta['phone']}")
                print(f"ID: {meta['candidate_id']}")
                return meta
        
        print("No candidate found for this phone number.")
        return None
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

if __name__ == "__main__":
    model, pinecone_index = ingest_candidates(JSON_PATH)
    
    # # Test the phone lookup functionality
    # test_phone = input("\nEnter a phone number to test lookup (or press Enter to skip): ").strip()
    # if test_phone:
    #     test_phone_lookup(model, pinecone_index, test_phone)