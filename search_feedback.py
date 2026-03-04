import sys
import chromadb

def search(query='product quality and freshness'):
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="customer_feedback")
    except Exception as e:
        print(f"Error initializing DB: {e}")
        return
        
    print(f"Collection contains {collection.count()} item vectors.")
    print(f"Executing semantic search for: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=20
    )
    
    for idx, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"\n--- Result {idx+1} (Distance: {dist:.4f}) ---")
        print(f"Document: {doc}")
        print(f"Metadata: {meta}")

if __name__ == '__main__':
    query = sys.argv[1] if len(sys.argv) > 1 else 'product quality and delivery'
    search(query)
