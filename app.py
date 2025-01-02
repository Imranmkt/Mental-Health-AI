from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from model.model_handler import ModelHandler
from model.ollama_embeddings import OllamaEmbeddingsWithMemory

app = Flask(__name__)
app.secret_key = 'local_development_key'  
CORS(app)  

# Initialize the ModelHandler and OllamaEmbeddings
model_handler = ModelHandler(model_name='mental_health_ai')  
ollama_embeddings = OllamaEmbeddingsWithMemory(model_name="mental_health_ai")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')

    # Check if it's a new conversation (first API call after a page refresh)
    if 'conversation_started' not in session:
        model_handler.memory.clear()  
        session['conversation_started'] = True 

    try:
        # You can choose whether to use the model_handler or ollama_embeddings based on the need
        # If you want to use ModelHandler for conversation history handling
        # model_response = model_handler.get_response(user_input)

        # If you want to use OllamaEmbeddings for context-based retrieval system
        model_response = ollama_embeddings.get_response(user_input)

        return jsonify({'response': model_response})

    except Exception as e:
        return jsonify({'response': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
