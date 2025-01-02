from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from rank_bm25 import BM25Okapi
from langchain_ollama.llms import OllamaLLM
from model.clean_ai_response import clean_response

# Load the dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train").select(range(100))
contexts = dataset["Context"]
responses = dataset["Response"]

# Combine context and responses
contexts_and_responses = [f"Context: {context} Response: {response}" for context, response in zip(contexts, responses)]

# Initialize memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history")

# Function to retrieve context using both dense and sparse retrieval
def retrieve_context(query, vectorstore, documents, k=3, alpha=0.7):
    # Step 1: Dense Retrieval
    dense_results = vectorstore.similarity_search_with_score(query, k=k)

    # Step 2: Sparse Retrieval (BM25)
    tokenized_docs = [doc['Context'].split() + doc['Response'].split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    sparse_scores = bm25.get_scores(query.split())

    # Step 3: Combine Scores (Dense + Sparse)
    dense_docs, dense_scores = zip(*dense_results)
    combined_scores = [(alpha * dense_scores[i] + (1 - alpha) * sparse_scores[i], dense_docs[i])
                       for i in range(len(dense_docs))]

    # Step 4: Rank by combined score (descending order)
    combined_scores.sort(key=lambda x: x[0], reverse=True)

    # Step 5: Return context and response for top k documents
    result = []
    for _, doc in combined_scores[:k]:
        context_response = doc.page_content.split("Response:")
        if len(context_response) == 2:
            context = context_response[0].replace("Context:", "").strip()
            response = context_response[1].strip()
            if any(emotion in context.lower() for emotion in ["sad", "anxious", "depressed", "not feeling good"]):
                result.append((context, response))
    return result

# Recognize intent (e.g., emotion, greeting)
def recognize_intent(user_query):
    user_query_lower = user_query.lower()
    intents = []

    if any(greet in user_query_lower for greet in ["hi", "hello", "hey"]):
        intents.append("greeting")
    if any(emotion in user_query_lower for emotion in ["sad", "anxious", "angry", "lonely", "depressed"]):
        intents.append("emotion")
    if any(bye_word in user_query_lower for bye_word in ["bye", "goodbye", "see you", "take care"]):
        intents.append("goodbye")

    return intents or ["query"]

# Generate the prompt with context and history
def generate_prompt(user_query, retrieved_context=None, chat_history=""):
    mental_health_prompt = (
        "You are a compassionate mental health assistant. Respond briefly, empathetically, and offer guidance when appropriate. "
        "Avoid lengthy explanations. Keep your responses short, kind, and supportive."
    )

    if not retrieved_context:
        # No relevant context, generate a general prompt
        return (
            f"{mental_health_prompt}\n"
            f"User Query: {user_query}\n"
            "Respond concisely with empathy, offering brief support or resources when needed."
        )

    # If context is found, include it in the prompt
    context = retrieved_context[0][0]
    response_text = retrieved_context[0][1]
    return (
        f"{mental_health_prompt}\n"
        f"Relevant Context: {context}\n"
        f"User Query: {user_query}\n"
        f"Relevant Response: {response_text}\n"
        "Respond empathetically, offering emotional support, provide resources, or next steps when the user indicates interest."
    )

# OllamaEmbeddings Class with integrated memory management and context handling
class OllamaEmbeddingsWithMemory:
    def __init__(self, model_name):
        # Initialize embeddings and language model only once
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)  # Pass model_name here
        self.vectorstore = FAISS.from_texts(contexts_and_responses, self.embeddings)
        self.memory = memory  # Conversation memory

    def get_response(self, user_input):
        # Update memory with the latest user input
        self.memory.save_context({"input": user_input}, {"output": ""})

        # Get the conversation history from memory
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # Retrieve relevant context using the dense and sparse retrieval mechanism
        retrieved_context = retrieve_context(user_input, self.vectorstore, dataset)

        if retrieved_context:
            post_prompt = generate_prompt(user_input, retrieved_context, chat_history)
        else:
            post_prompt = generate_prompt(user_input, chat_history=chat_history)

        try:
            # Get response from the model
            response = self.llm.invoke([{"role": "user", "content": post_prompt}])

            # Clean and return the model response
            model_response = response.get('message', {}).get('content', '').strip() if not isinstance(response, str) else response.strip()
            cleaned_response = clean_response(model_response)
            return cleaned_response
        except Exception as e:
            raise Exception(f"Error contacting the model: {str(e)}")

# Example usage
# ollama_handler = OllamaEmbeddingsWithMemory(model_name="mental_health_ai")
# user_input = "I'm feeling anxious and overwhelmed, what can I do?"
# response = ollama_handler.get_response(user_input)
# print("Response:", response)
