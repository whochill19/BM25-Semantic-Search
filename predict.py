import re
import pandas as pd
import difflib

from bm25_model import BM25
from preprocess import preprocess_query, highlight_text, display_search_results_clean
from embedding_model import load_embedding, hybrid_search

df = pd.read_csv('./dataset/processed_documents.csv')

bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(df['processed_document'].tolist())

try:
    model, corpus_embeddings = load_embedding('./dataset/models')
    use_semantic = True
except Exception as e:
    print(f"⚠️ Tidak dapat memuat model embedding ({e}), fallback ke BM25 saja.")
    model, corpus_embeddings = None, None
    use_semantic = False

def correct_spelling(query, vocab, cutoff=0.8):
    words = query.lower().split()
    corrected_words = []

    for word in words:
        matches = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)
        if matches:
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)

    corrected_query = " ".join(corrected_words)
    return corrected_query if corrected_query != query else None

print("\nMEDICINE SEARCH ENGINE")
print("Cari informasi obat atau gejala umum.")
print("Ketik 'quit' untuk keluar.\n")

while True:
    try:
        query = input("Cari: ").strip()

        if query.lower() in ['quit', 'exit', 'keluar', 'q']:
            print("\nTerima kasih telah menggunakan Medicine Search Engine.")
            break

        if not query:
            print("Masukkan kata kunci pencarian.")
            continue

        # Preprocess query
        processed_query = preprocess_query(query)

        # Pencarian dengan BM25
        results = bm25.search(processed_query, top_k=10)

        if results:
            display_search_results_clean(results, query, df)

        # Kalau BM25 nggak nemu, fallback ke Semantic Embedding
        elif use_semantic:
            print(f"\nBM25 tidak menemukan hasil relevan untuk: '{query}'")
            print("Mencoba pencarian berbasis makna (Semantic Embedding)...\n")

            # Preprocessing query
            processed_query = preprocess_query(query)

            # Ambil seluruh vocabulary dari corpus
            corpus_vocab = set(" ".join(df['processed_document']).split())

            # Coba koreksi ejaan query
            corrected = correct_spelling(query, corpus_vocab)

            if corrected:
                print(f"Menampilkan hasil untuk: '{corrected}'")
                print(f"(Pencarian awal: '{query}')\n")
                query_to_use = corrected
            else:
                query_to_use = processed_query

            results = hybrid_search(
                query=query_to_use,
                model=model,
                corpus_embeddings=corpus_embeddings,
                bm25=bm25,
                df=df,
                alpha=0.6,
                top_k=10
            )

            if not results:
                print(f"\nTidak ditemukan hasil untuk: {query}")
                print("Coba gunakan kata yang lebih umum.\n")
            else:
                display_search_results_clean(results, query_to_use, df)

            # hybrid_results = hybrid_search(
            #     query=query_to_use,
            #     model=model,
            #     corpus_embeddings=corpus_embeddings,
            #     bm25=bm25,
            #     df=df,
            #     alpha=0.6,
            #     top_k=10
            # )

            # if not hybrid_results:
            #     print("Tidak ada hasil hybrid.\n")
            #     continue

            # print(f"\nHasil pencarian untuk: {query}")
            # display_search_results_clean(hybrid_results, query, df)

    except KeyboardInterrupt:
        print("\nTerima kasih telah menggunakan Medicine Search Engine.")
        break
    except Exception as e:
        print(f"Terjadi kesalahan: {e}\n")