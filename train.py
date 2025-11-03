import sys
import pandas as pd
import numpy as np
import os
from bm25_model import BM25
from embedding_model import train_embedding
from preprocess import preprocess
from data_loader import load_dataset

DATA_PATH = './dataset/Medicine_Details.csv'
PROCESSED_PATH = './dataset/processed_documents.csv'
MODEL_DIR = './dataset/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_bm25(df):
    print("\n=== Training BM25 Model ===")
    documents = df['processed_document'].tolist()

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(documents)

    # Simpan model BM25 sederhana (idf & param)
    np.save(os.path.join(MODEL_DIR, 'bm25_idf.npy'), bm25.idf)
    np.save(os.path.join(MODEL_DIR, 'bm25_vocab.npy'), list(bm25.vocab))
    print("✅ BM25 model trained and saved successfully!")
    print(f"Vocabulary size: {len(bm25.vocab)} terms")
    print(f"Average document length: {bm25.avg_doc_length:.1f} words")
    return bm25


def train_semantic(df):
    print("\n=== Training Semantic Embedding ===")
    train_embedding(df, model_name='all-MiniLM-L6-v2', save_dir=MODEL_DIR)
    print("✅ Semantic Embedding trained and saved successfully!")

def ensure_preprocessed():
    """Pastikan dataset sudah diproses"""
    if not os.path.exists(PROCESSED_PATH):
        print("⚠️ Processed dataset belum ditemukan. Melakukan preprocess...")
        df = load_dataset(DATA_PATH)
        df = preprocess(df)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"✅ Dataset berhasil diproses dan disimpan di {PROCESSED_PATH}")
    else:
        df = pd.read_csv(PROCESSED_PATH)
    return df

def main():
    # Pastikan dataset sudah siap
    df = ensure_preprocessed()

    # Kalau user gak kasih argumen, tampilkan panduan
    if len(sys.argv) < 2:
        print("⚙️ Gunakan salah satu perintah berikut:")
        print("   python train.py bm25       -> latih model BM25 saja")
        print("   python train.py embedding  -> latih model Semantic Embedding saja")
        print("   python train.py all        -> latih BM25 dan Embedding sekaligus")
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == '-bm25':
        train_bm25(df)

    elif mode == '-embedding':
        train_semantic(df)

    elif mode == 'all':
        print("\n🚀 Training BM25 dan Semantic Embedding sekaligus...")
        bm25 = train_bm25(df)
        train_semantic(df)
        print("\n✅ Semua model berhasil dilatih dan disimpan!")

    else:
        print(f"❌ Mode '{mode}' tidak dikenal. Gunakan 'bm25', 'embedding', atau 'all'.")


if __name__ == "__main__":
    main()