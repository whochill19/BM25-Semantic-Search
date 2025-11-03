# preprocess.py
import re
import pandas as pd
import nltk
from data_loader import load_dataset

# === Unduh resource NLTK (sekali aja) ===
def ensure_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# === Text cleaning dasar ===
def preprocess_text(text):
    """Bersihkan teks: lowercase + buang tanda baca + normalisasi spasi."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # hanya huruf, angka, dan spasi
    text = re.sub(r'\s+', ' ', text)      # hapus spasi berlebih
    return text.strip()

# === Proses utama dataset ===
def preprocess(df, text_columns=None, verbose=True):
    """
    Gabungkan kolom teks utama + buat kolom hasil cleaning.
    """
    if text_columns is None:
        text_columns = ['Medicine Name', 'Uses', 'Composition', 'Side_effects']

    if verbose:
        print("=== INFO DATASET ===")
        print(df.info())
        print("\nJumlah data missing per kolom:")
        print(df.isnull().sum())
        print("=" * 40)

    # Pastikan kolom teks tersedia
    available_columns = [col for col in text_columns if col in df.columns]
    if not available_columns:
        raise ValueError(f"Tidak ada kolom teks yang ditemukan. Dicari: {text_columns}")

    if verbose:
        print("Kolom yang akan digabung:", available_columns)

    # Gabungkan kolom teks
    df['combined_document'] = df[available_columns].fillna('').apply(
        lambda row: ' '.join(row.values.astype(str)),
        axis=1
    )

    # Tambahkan kolom hasil preprocessing
    df['processed_document'] = df['combined_document'].apply(preprocess_text)

    if verbose:
        avg_len = df['processed_document'].str.len().mean()
        print(f"✅ Preprocessing completed! Jumlah dokumen: {len(df)}, panjang rata-rata: {avg_len:.0f} karakter")

    return df

def highlight_text(text, query_terms):
    for term in query_terms:
        if term.lower() in text.lower():
            text = re.sub(
                f'({re.escape(term)})',
                '**\\1**',       # pakai markdown bold-style
                text,
                flags=re.IGNORECASE
            )
    return text

def preprocess_query(query):
    query = str(query).lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def display_search_results_clean(results, query, df):
    """Tampilkan hasil pencarian gaya minimalis (tanpa emoji)"""
    print(f"\nHasil pencarian untuk: {query}")
    print(f"{len(results)} hasil ditemukan.\n")

    query_terms = query.lower().split()

    for rank, (doc_index, score) in enumerate(results, 1):
        row = df.iloc[doc_index]
        medicine_name = row.get('Medicine Name', 'Tidak tersedia')
        uses = row.get('Uses', 'Tidak tersedia') or "Tidak tersedia"
        composition = row.get('Composition', 'Tidak tersedia') or "Tidak tersedia"
        side_effects = row.get('Side_effects', 'Tidak tersedia') or "Tidak tersedia"

        highlighted_uses = highlight_text(str(uses), query_terms)
        highlighted_composition = highlight_text(str(composition), query_terms)

        print(f"{rank}. {medicine_name}  (score: {score:.4f})")
        print(f"   Kegunaan     : {highlighted_uses}")
        print(f"   Komposisi    : {highlighted_composition}")
        print(f"   Efek samping : {side_effects}\n")

# === Jika dijalankan langsung (bukan diimpor) ===
if __name__ == "__main__":
    ensure_nltk_resources()
    path = './dataset/Medicine_Details.csv'   # path dataset asli
    df = load_dataset(path)
    df = preprocess(df)

    # Simpan SEMUA kolom, bukan cuma processed
    df.to_csv('./dataset/processed_documents.csv', index=False)
    print("✅ Full dataset (with processed columns) saved to ./dataset/processed_documents.csv")
