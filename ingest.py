import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Local file path for JSON
JSON_PATH = "dummy-resume.json"

# Global metadata store: maps FAISS index IDs to metadata
metadata_store = []

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

def setup_faiss(dim=384):
    return faiss.IndexFlatL2(dim)

def process_candidate(model, faiss_index, candidate_id, candidate_data):
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

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).astype("float32")
        faiss_index.add(np.array([embedding]))

        metadata_store.append({
            "candidate_id": candidate_id,
            "name": name,
            "phone": phone_number,
            "chunk_id": i,
            "text": chunk,
            "is_phone_entry": False
        })

    # Optional: store phone number as separate searchable item
    phone_query = f"Phone number: {phone_number}"
    phone_embedding = model.encode(phone_query).astype("float32")
    faiss_index.add(np.array([phone_embedding]))

    metadata_store.append({
        "candidate_id": candidate_id,
        "name": name,
        "phone": phone_number,
        "chunk_id": None,
        "text": phone_query,
        "is_phone_entry": True
    })

    return True

def ingest_candidates(json_path):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Setting up FAISS...")
    faiss_index = setup_faiss()

    with open(json_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    if isinstance(candidates, dict):  
        candidates = list(candidates.values())

    print(f"Found {len(candidates)} candidates")

    success_count = 0
    for i, candidate_data in tqdm(enumerate(candidates), desc="Processing candidates"):
        candidate_id = f"candidate_{i+1}"
        if process_candidate(model, faiss_index, candidate_id, candidate_data):
            success_count += 1

    print(f"Successfully processed {success_count} out of {len(candidates)} candidates")

    return model, faiss_index

def test_phone_lookup(model, faiss_index, phone_number):
    query = f"Phone number: {phone_number}"
    query_vector = model.encode(query).astype("float32").reshape(1, -1)

    D, I = faiss_index.search(query_vector, k=1)
    idx = I[0][0]

    if 0 <= idx < len(metadata_store):
        meta = metadata_store[idx]
        if meta.get("is_phone_entry"):
            print(f"Found candidate: {meta['name']} (ID: {meta['candidate_id']})")
            return meta
    print("No candidate found for this phone number.")
    return None

if __name__ == "__main__":
    model, faiss_index = ingest_candidates(JSON_PATH)