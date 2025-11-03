# main.py
from data_loader import load_dataset
from preprocess import preprocess_dataset
from BM25_model import BM25
from embedding_model import embedding_model, encode_documents, hybrid_search

# === 1. Load dataset ===
path = './dataset/Medicine_Details.csv'
df = load_dataset(path)

# === 2. Preprocessing ===
text_cols = ['Medicine Name', 'Uses', 'Composition', 'Side_effects']
df = preprocess_dataset(df, text_cols)

# === 3. BM25 model ===
bm25 = BM25().fit(df['processed_document'])

# === 4. Encode semantic embeddings ===
corpus_embeddings = encode_documents(df)

# === 5. Hybrid search ===
query = "obat untuk menurunkan tekanan darah"
results = hybrid_search(query, bm25, df, corpus_embeddings, alpha=0.6, top_k=5)

print("\n=== HASIL PENCARIAN ===")
print(results)
