from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv("config.env")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
print(os.environ.get('PINECONE_API_KEY'))

# Initialize your Ollama embeddings with your model
embeddings = OllamaEmbeddings(model="mental_health_ai")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mental-health-chat"

# Check embedding dimension
test_embedding = embeddings.embed_query("test input")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")

# Delete and recreate the index if needed
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=embedding_dim,  
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Access the index
index = pc.Index(index_name)

# Load and prepare dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(100))
contexts = dataset["Context"]
responses = dataset["Response"]

# Embed documents and upload to Pinecone
for i, context in enumerate(contexts):
    embedding = embeddings.embed_query(context)
    index.upsert([(f"doc-{i}", embedding, {"Response": responses[i]})])

def get_ai_response(user_input):
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(user_input)

    # Query the index with proper keyword arguments
    query_results = index.query(
        vector=query_embedding,  
        top_k=3,                 
        include_metadata=True    
    )

    # Check and return the best match
    if query_results.matches:
        best_match = query_results.matches[0]
        return best_match.metadata["Response"]
    else:
        return "Sorry, I couldn't find a suitable response."


# Example usage
user_input = "I'm feeling very anxious lately."
response = get_ai_response(user_input)
print(response)
