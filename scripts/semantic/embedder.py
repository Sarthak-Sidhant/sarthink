import json
import argparse
import os
import sys
from typing import List, Dict
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Warning: Please ensure lancedb, pyarrow, and sentence-transformers are installed.")

EMBEDDING_MODEL_NAME = "Qwen3-Embedding-4B"
LANCEDB_PATH = str(REPO_ROOT / "processed_data" / "graph" / "sarthink_lancedb")


def topic_chunk_burst(chunk: Dict, embedding_model) -> List[Dict]:
    topic_chunks = []
    text = chunk.get('text', '')
    
    micro_bursts = [b.strip() for b in text.split('\n\n') if b.strip()]
    
    if len(micro_bursts) <= 1:
        return []
        
    embeddings = embedding_model.encode(micro_bursts, convert_to_numpy=True)
    
    similarities = []
    for i in range(len(embeddings)-1):
        v1 = embeddings[i] / max(np.linalg.norm(embeddings[i]), 1e-10)
        v2 = embeddings[i+1] / max(np.linalg.norm(embeddings[i+1]), 1e-10)
        similarities.append(np.dot(v1, v2))
        
    threshold = 0.6 
    
    current_topic_bursts = [micro_bursts[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            topic_chunks.append({
                "parent_id": chunk.get('channel_id'),
                "start_time": chunk.get('start_time'),
                "end_time": chunk.get('end_time'),
                "density_score": chunk.get('density_score', 0.0),
                "ego_weight": chunk.get('ego_weight', 0.0),
                "text": "\n".join(current_topic_bursts)
            })
            current_topic_bursts = [micro_bursts[i+1]]
        else:
            current_topic_bursts.append(micro_bursts[i+1])
            
    if current_topic_bursts:
        topic_chunks.append({
            "parent_id": chunk.get('channel_id'),
            "start_time": chunk.get('start_time'),
            "end_time": chunk.get('end_time'),
            "density_score": chunk.get('density_score', 0.0),
            "ego_weight": chunk.get('ego_weight', 0.0),
            "text": "\n".join(current_topic_bursts)
        })
        
    return topic_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(REPO_ROOT / "processed_data" / "semantic" / "summarized_chunks.json"), help="Path to summarized JSON output")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            updated_chunks = json.load(f)
    except FileNotFoundError:
        print(f"{args.input} not found!")
        return

    print(f"Loading Embedding model: {EMBEDDING_MODEL_NAME} on GPU...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda", trust_remote_code=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load onto GPU, attempting CPU fallback: {e}")
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")

    print(f"Generating Topic Chunks and embedding vectors for {len(updated_chunks)} sessions...")
    
    db = lancedb.connect(LANCEDB_PATH)
    
    session_data_for_db = []
    topic_data_for_db = []

    for idx, chunk in enumerate(updated_chunks):
        if idx % 100 == 0:
            print(f"Embedded {idx}/{len(updated_chunks)} chunks...")
            
        session_text = f"Participants: {', '.join(chunk.get('participants', []))} \n {chunk.get('summary', '')}"
        session_vec = embedding_model.encode(session_text, convert_to_numpy=True).tolist()
        
        session_data_for_db.append({
            "vector": session_vec,
            "channel_id": chunk.get('channel_id', ''),
            "platform": chunk.get('platform', ''),
            "title": chunk.get('title', ''),
            "start_time": chunk.get('start_time', ''),
            "end_time": chunk.get('end_time', ''),
            "density_score": chunk.get('density_score', 0.0),
            "ego_weight": chunk.get('ego_weight', 0.0),
            "summary": chunk.get('summary', ''),
            "text": chunk.get('text', '') 
        })
        
        sub_chunks = topic_chunk_burst(chunk, embedding_model)
        for t_chunk in sub_chunks:
            topic_vec = embedding_model.encode(t_chunk['text'], convert_to_numpy=True).tolist()
            topic_data_for_db.append({
                "vector": topic_vec,
                "parent_id": t_chunk['parent_id'],
                "start_time": t_chunk['start_time'],
                "end_time": t_chunk['end_time'],
                "density_score": t_chunk['density_score'],
                "ego_weight": t_chunk['ego_weight'],
                "text": t_chunk['text']
            })

    print("\nSaving tables to LanceDB...")
    
    if "sessions" in db.table_names():
        db.drop_table("sessions")
    db.create_table("sessions", data=session_data_for_db)
    
    if "topics" in db.table_names():
        db.drop_table("topics")
    if topic_data_for_db:
        db.create_table("topics", data=topic_data_for_db)

    print("Vector database successfully populated.")
    print(f"Ingested {len(session_data_for_db)} session chunks and {len(topic_data_for_db)} dynamic topic chunks.")

if __name__ == "__main__":
    main()
