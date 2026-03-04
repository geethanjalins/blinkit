import pandas as pd
import chromadb
import sys
import gc

def init_vector_db():
    print("Loading Customer Feedback File...")
    try:
        df = pd.read_csv('customer.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Check for required fields
    required_columns = {'feedback_id', 'feedback_category', 'sentiment', 'feedback_text'}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"Missing required columns in CSV: {missing}")
        return

    # Drop missing feedback text rows
    df = df.dropna(subset=['feedback_text'])
    
    documents = []
    metadatas = []
    ids = []
    
    print(f"Total valid feedback entries: {len(df)}")
    print("Pre-processing text items into embeddings payload...")
    
    # Iterate dynamically
    for idx, row in df.iterrows():
        cat = str(row['feedback_category']).strip()
        sent = str(row['sentiment']).strip()
        txt = str(row['feedback_text']).strip()
        fbid = str(row['feedback_id'])
        
        # We ensure meaning related to product quality, items, freshness, delivery, experience is captured
        # by simply fusing all aspects with clear textual headers. 
        # Since chroma uses 'all-MiniLM-L6-v2', meaning extraction from such textual statements is highly effective.
        combined_text = f"Topic: {cat}. Sentiment: {sent}. Feedback Context: {txt}"
        documents.append(combined_text)
        
        # Maintain metadata for easy filtering 
        metadatas.append({
            "category": cat,
            "sentiment": sent,
            "feedback_id": fbid
        })
        
        # ID must be unique string
        uid = f"fb_{fbid}_{idx}"
        ids.append(uid)
        
    print("Initializing ChromaDB Persistent Client...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create our feedback index
    # We allow chroma default sentence-transformers embedding
    collection = client.get_or_create_collection(name="customer_feedback")
    
    # Upsert data in chunks as chroma API restricts batch size
    batch_size = 5000 
    
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"Upserting batch {i} to {end}...")
        collection.upsert(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
    
    print("\n[✔] Database successfully populated!")
    print(f"Total entries in vector db: {collection.count()}")

if __name__ == "__main__":
    init_vector_db()
