import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import pickle


def train_embedding(df, model_name='all-MiniLM-L6-v2', save_dir='./dataset/models'):
    """
    Melatih Semantic Embedding berdasarkan kolom 'processed_document'.
    Menyimpan model dan embedding ke folder models.
    """
    print("\n=== Training Semantic Embedding ===")
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = SentenceTransformer(model_name)
    print(f"Loaded model: {model_name}")

    # Encode semua dokumen
    texts = df['processed_document'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Simpan embedding ke file
    emb_path = os.path.join(save_dir, 'corpus_embeddings.npy')
    np.save(emb_path, embeddings)

    # Simpan nama model agar tahu model apa yang dipakai nanti
    with open(os.path.join(save_dir, 'embedding_model_name.pkl'), 'wb') as f:
        pickle.dump(model_name, f)

    print(f"✅ Embedding disimpan di: {emb_path}")
    return model, embeddings

def load_embedding(model_dir='./dataset/models'):
    """
    Muat model SentenceTransformer + embedding corpus.
    """
    print("Memuat model Semantic Embedding...")
    model_name = pickle.load(open(os.path.join(model_dir, 'embedding_model_name.pkl'), 'rb'))
    model = SentenceTransformer(model_name)
    corpus_embeddings = np.load(os.path.join(model_dir, 'corpus_embeddings.npy'))
    print(f"✅ Model '{model_name}' dan embedding berhasil dimuat!")
    return model, corpus_embeddings

def semantic_search(query, model, corpus_embeddings, df, top_k=5):
    """
    Cari dokumen paling relevan secara semantik.
    """
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    # ambil indeks top-k
    top_indices = np.argsort(-cosine_scores)[:top_k]

    # results = df.iloc[top_indices][['Medicine Name', 'Uses', 'Composition', 'Side_effects']].copy()
    # results['Score'] = cosine_scores[top_indices]
    # return results
    return [(int(i), float(cosine_scores[i])) for i in top_indices]

def hybrid_search(query, bm25, df, corpus_embeddings, alpha=0.6, top_k=10, model=None):
    """
    Hybrid search: gabung skor BM25 dan Semantic Embedding.
    alpha menentukan bobot BM25 (0.0–1.0).
    """
    # Encode query ke embedding
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    semantic_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

    # Hitung skor BM25 manual
    query_terms = query.lower().split()
    bm25_scores = []
    for doc in df['processed_document']:
        s = 0
        for term in query_terms:
            if term in bm25.idf:
                tf = doc.split().count(term)
                dl = len(doc.split())
                s += bm25.idf[term] * (tf * (bm25.k1 + 1) /
                                       (tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avg_doc_length)))
        bm25_scores.append(s)

    # Normalisasi skor biar setara
    bm25_scores = np.array(bm25_scores)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)

    # Gabungkan skor
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * semantic_scores

    # Ambil top-k
    top_indices = np.argsort(-hybrid_scores)[:top_k]
    results = [(int(i), float(hybrid_scores[i])) for i in top_indices]

    return results

def smart_search(query, bm25, df, corpus_embeddings, model, threshold=0.1, top_k=5):
    """
    Kombinasi: BM25 → fallback ke Semantic Embedding kalau skor rendah.
    """
    query_tokens = query.lower().split()

    # Skor BM25 manual
    bm25_scores = []
    for doc in df['processed_document']:
        s = 0
        for term in query_tokens:
            if term in bm25.idf:
                tf = doc.split().count(term)
                dl = len(doc.split())
                s += bm25.idf[term] * (tf * (bm25.k1 + 1) / (tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avg_doc_length)))
        bm25_scores.append(s)

    max_bm25 = max(bm25_scores)
    print(f"\nBM25 skor tertinggi: {max_bm25:.4f}")

    # kalau BM25 bagus, pakai hasilnya
    if max_bm25 > threshold:
        print("Menggunakan hasil dari BM25 (lexical match)")
        top_indices = np.argsort(-np.array(bm25_scores))[:top_k]
        results = df.iloc[top_indices][['Medicine Name', 'Uses', 'Composition', 'Side_effects']].copy()
        results['Score'] = np.array(bm25_scores)[top_indices]
        return results

    # kalau BM25 lemah, fallback ke semantic
    print("Fallback ke Semantic Embedding (makna mirip)")
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    top_indices = np.argsort(-cosine_scores)[:top_k]

    results = df.iloc[top_indices][['Medicine Name', 'Uses', 'Composition', 'Side_effects']].copy()
    results['Score'] = cosine_scores[top_indices]
    return results

