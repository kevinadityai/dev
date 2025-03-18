import time
import gradio as gr

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

import qdrant_client
from qdrant_client.models import Distance

from sentence_transformers import CrossEncoder

# Configuration parameters.
url = "http://localhost:6333"
collection_name = "faq-question"  
dimension = 768  
distance = Distance.COSINE
top_n = 20

# Initiate FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings(
    cache_dir="../embedding_cache",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Initialize GTE multilingual reranker
reranker = CrossEncoder('Alibaba-NLP/gte-multilingual-reranker-base', trust_remote_code=True)
client = qdrant_client.QdrantClient(url=url)

def search_faq(query: str, mode: str) -> str:
    """
    Mencari FAQ berdasarkan query dengan dua mode:
      - "Normal": melakukan vector search dengan limit 20 dan menampilkan seluruh hasil.
      - "Reranker": melakukan vector search, mengurutkan ulang hasilnya dengan cross-encoder reranker,
                    dan hanya menampilkan 1 hasil terbaik.
    
    Waktu eksekusi keseluruhan query juga ditampilkan.
    """
    start_time = time.time()

    # Embed query.
    query_embedding = embeddings.embed_query(query)
    
    # Lakukan vector search dengan limit 20.
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_n
    )
    
    # Jika mode reranker, lakukan re-ranking dan ambil hanya hasil terbaik.
    if mode == "Reranker" and search_results:
        # Buat pasangan [query, candidate_question] untuk setiap hasil.
        rerank_pairs = []
        for result in search_results:
            metadata = result.payload.get("metadata", {})
            question_text = metadata.get("question", "N/A")
            rerank_pairs.append([query, question_text])
        
        # Dapatkan skor reranker untuk setiap pasangan.
        scores = reranker.predict(rerank_pairs)
        # Gabungkan hasil pencarian dengan skor reranker dan urutkan secara menurun.
        results_with_scores = list(zip(search_results, scores))
        results_with_scores = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
        # Ambil hanya hasil terbaik.
        search_results = [results_with_scores[0][0]]
    
    elapsed_time = time.time() - start_time

    # Bangun output Markdown.
    if not search_results:
        output = f"### Query Time: {elapsed_time:.4f} seconds\n\n"
        output += "No matching FAQs found. Please try a different question."
        return output

    output = f"### Query Time: {elapsed_time:.4f} seconds\n\n"
    
    if mode == "Normal":
        output += "### Top Matching FAQs (All 20 Results):\n\n"
    else:
        output += "### Top Matching FAQ (Best Result):\n\n"
    
    for i, result in enumerate(search_results, start=1):
        metadata = result.payload.get("metadata", {})
        faq_question = metadata.get("question", "N/A")
        faq_answer = metadata.get("answer", "N/A")
        faq_category = metadata.get("category", "N/A")
        
        output += f"**{i}. Question:** {faq_question}\n\n"
        output += f"**Answer:** {faq_answer}\n\n"
        output += f"**Category:** {faq_category}\n\n"
        output += "---\n\n"
    return output

# Gradio interface.
iface = gr.Interface(
    fn=search_faq,
    inputs=[
        gr.Textbox(label="Enter your question", placeholder="Type your question here..."),
        gr.Radio(choices=["Normal", "Reranker"], label="Search Mode", value="Normal")
    ],
    outputs=gr.Markdown(),
    title="Hospital FAQ Search",
    description="Enter a question and select the search mode to display matching FAQ entries along with the query time."
)

if __name__ == "__main__":
    iface.launch()
