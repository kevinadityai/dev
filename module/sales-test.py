import qdrant_client
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance
from dotenv import load_dotenv
import gradio as gr

# Load environment variables.
load_dotenv()

# Initialize OpenAI embeddings.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
)

# Configuration parameters.
url = "http://localhost:6333"
collection_name = "test_salesItem"
distance = Distance.COSINE
dimension = 1536 
top_n = 10

# Initialize the Qdrant client.
client = qdrant_client.QdrantClient(url=url)

def search(query: str, formularium: str) -> str:
    """
    Takes a text query and a selectable Formularium filter,
    converts the query to an embedding, searches the Qdrant collection,
    and returns a formatted Markdown string with the top matching items.
    """
    # Embed the query.
    query_embedding = embeddings.embed_documents([query])[0]
    
    # Build the query filter if a specific Formularium is selected.
    query_filter = None
    if formularium != "All":
        query_filter = {
            "must": [
                {
                    "key": "metadata.Formularium",
                    "match": {
                        "value": formularium
                    }
                }
            ]
        }
    
    # Perform the vector search with an optional filter.
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_n,
        query_filter=query_filter
    )
    
    # Build a user-friendly Markdown output.
    if not search_results:
        return "No matching items were found. Please try a different query or filter."
    
    output = "### Top Matching Items:\n\n"
    for i, result in enumerate(search_results, start=1):
        metadata = result.payload.get("metadata", {})
        sales_item_name = metadata.get("SalesItemName", "N/A")
        sales_item_code = metadata.get("SalesItemCode", "N/A")
        active_ingredients = metadata.get("ActiveIngredientsName", "N/A")
        formularium_value = metadata.get("Formularium", "N/A")
        
        output += f"**{i}. Item Name:** {sales_item_name}\n\n"
        output += f"- **Item Code:** {sales_item_code}\n"
        output += f"- **Active Ingredients:** {active_ingredients}\n"
        output += f"- **Formularium:** {formularium_value}\n"
        output += f"- **Relevance Score:** {result.score:.4f}\n\n"
    return output

# Create a Gradio interface with a textbox for the query and a radio button for the Formularium filter.
iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Enter your search query", placeholder="Type your query here..."),
        gr.Radio(choices=["All", "NORMAL", "BPJS", "INHEALTH"],
                 label="Select Formularium Filter",
                 value="All")
    ],
    outputs=gr.Markdown(),
    title="Sales Items Search",
    description="Enter a query and select a Formularium filter to search for sales items in the vector database. Results are ranked based on relevance."
)

if __name__ == "__main__":
    iface.launch()
