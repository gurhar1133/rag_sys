import re
import glob
import json
import os
import pandas as pd
from pathlib import Path
import torch
import numpy as np
from openai import OpenAI
import sentence_transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



_MODEL_PATH = "all-MiniLM-L6-v2"
_RERANKER_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DOCS_PATH = "./docs"


def query_rewrite(query):
    if not Path("secrets.json").exists():
        return query

    with open("secrets.json") as f:
        secrets = json.load(f)
    client = OpenAI(api_key=secrets["OPENAI_API_KEY"])
    prompt = (
            "The following text is a query for book search retrieval"
            "Please rewrite queries such that if the user says 'I want somethin like x' that a description of x is substituded for the title. This is in order to make the query more robust"
            "Additionally if the user enters a query not related to book retrieval, I want your response to be 'query not relevant'"
            f"Text:\n{query}"
        )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a book search query rewriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return resp.choices[0].message.content.strip()


def download_models():
    #if model path exists, don't download
    if not os.path.exists(_RERANKER_PATH):
        reranker = sentence_transformers.CrossEncoder(_RERANKER_PATH)
        reranker.save(_RERANKER_PATH)
    if not os.path.exists(_MODEL_PATH):
        model = sentence_transformers.SentenceTransformer(_MODEL_PATH)
        model.save(_MODEL_PATH)


def vector_model():
    model = SentenceTransformer(
        _MODEL_PATH,
    )
    return model


def reranker_model():
    reranker = CrossEncoder(
        _RERANKER_PATH,
    )
    return reranker


def simple_chunker(text, chunk_size, overlap):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)

        # small heuristic: try to end on newline / sentence boundary
        window = text[i:j]
        cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if cut > 200 and j < n:   # avoid cutting too early
            j = i + cut + 1

        chunk = text[i:j].strip()
        if chunk:
            chunks.append((chunk, i, j))
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def construct_vector_db(path, emb_model, chunk_size=100, overlap=20):
    # load docs
    docs = ingest_txt_docs(path)
    chunk_texts = []
    meta = []

    for d in docs:
        doc_id = d["doc_id"]
        text = d["content"]

        for chunk_id, (chunk, start, end) in enumerate(simple_chunker(text, chunk_size, overlap)):
            chunk_texts.append(chunk)
            meta.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end,
                "chunk_text": chunk,
            })

    vdb = emb_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    ).astype("float32")


    tfidf = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf.fit_transform(chunk_texts) 

    return vdb, meta, chunk_texts, tfidf, X_tfidf


def ingest_txt_docs(path, max_docs=3):
    search_pattern = os.path.join(path, '*.txt')
    file_list = glob.glob(search_pattern)
    
    file_contents = []
    for idx, filename in enumerate(file_list):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            if len(text) > 20: # simple doc size handler
                file_contents.append({"doc_id": filename, "content": text})
    # for f in file_contents:
    #     print(f)
    # assert 0
    return file_contents


def test_system(test_query, k=6):
    query_original = test_query
    test_query = query_rewrite(test_query)
    if "query not relevant" in test_query.lower():
        print("I do not know")
        return
    download_models()
    emb_model = vector_model()
    reranker = reranker_model()

    vdb, chunk_meta, chunk_texts, tfidf, X_tfidf = construct_vector_db(_DOCS_PATH, emb_model)
    
    q_emb = emb_model.encode([test_query], normalize_embeddings=True)
    vdb_scores = cosine_similarity(q_emb, vdb)[0]
    vdb_idx = np.argsort(vdb_scores)[-int(k/2):][::-1]

    
    q_t = tfidf.transform([test_query])             
    # tfidf_scores = cosine_similarity(q_t, X_tfidf)[0]
    tfidf_scores = (X_tfidf @ q_t.T).toarray().ravel()
    tfidf_idx = np.argsort(tfidf_scores)[-int(k/2):][::-1]

    # NAIVE UNION FOR NOW
    seen = set()
    hybrid_idx = []
    for i in vdb_idx:
        if int(i) not in seen:
            hybrid_idx.append(int(i)); seen.add(int(i))
    for i in tfidf_idx:
        if int(i) not in seen:
            hybrid_idx.append(int(i)); seen.add(int(i))

    hybrid_idx = hybrid_idx[:k]

    pairs = [(test_query, chunk_texts[i]) for i in hybrid_idx]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(hybrid_idx, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )
    reranked_idx = [i for i, _ in reranked]
    hits = []
    for i in reranked_idx:
        m = chunk_meta[i]
        # Answerability gating start
        if vdb_scores[i] > .2 and tfidf_scores[i] > .0:
            hits.append({
                "idx": i,
                "doc_id": m["doc_id"],
                "chunk_id": m["chunk_id"],
                "start_char": m["start_char"],
                "end_char": m["end_char"],
                "vec_score": float(vdb_scores[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "excerpt": m["chunk_text"],
            })

    if len(hits) < 1:
        print("I don't know")
        return 
    
    print("\nYOUR QUERY:", query_original)
    print("\nRESULTS:\n")
    for hit in hits:
        print("DOCUMENT:", hit["doc_id"])
        print()
        print("EVIDENCE EXCERPT:", hit["excerpt"])
        print()
        print("CHUNK ID:", hit["chunk_id"])
        print("-"*50)


if __name__ == "__main__":
    for i in range(3):
        query = Path(f"./queries/query{str(i + 1)}.txt").read_text()
        test_system(query)