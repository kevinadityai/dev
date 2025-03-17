import qdrant_client
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance
from dotenv import load_dotenv
import gradio as gr

# Load environment variables.
load_dotenv()

# Configuration parameters.
url = "http://localhost:6333"
collection_name = "test_faq"  
dimension = 1536  
distance = Distance.COSINE
top_n = 3

# Initialize the embeddings and Qdrant client.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = qdrant_client.QdrantClient(url=url)

def search_faq(query: str) -> str:
    """
    Takes a question, converts it to an embedding, searches the Qdrant collection,
    and returns a formatted Markdown string with the top matching FAQ entries.
    """
    # Embed the query.
    query_embedding = embeddings.embed_query(query)
    
    # Perform the vector search.
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_n
    )
    
    # Build a user-friendly Markdown output.
    if not search_results:
        return "No matching FAQs found. Please try a different question."
    
    output = "### Top Matching FAQs:\n\n"
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

# Create a Gradio interface.
iface = gr.Interface(
    fn=search_faq,
    inputs=gr.Textbox(label="Enter your question", placeholder="Type your question here..."),
    outputs=gr.Markdown(),
    title="Hospital FAQ Search",
    description="Enter a question to search for matching FAQ entries."
)

if __name__ == "__main__":
    iface.launch()
