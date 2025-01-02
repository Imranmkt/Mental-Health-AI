from model.clean_ai_response import clean_response
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory


class ModelHandler:
    def __init__(self, model_name):
        # Initialize the language model and memory
        self.llm = OllamaLLM(model=model_name)
        self.memory = ConversationBufferMemory(memory_key="chat_history")



    def get_response(self, user_input):
        # Update memory with the latest user input
        self.memory.save_context({"input": user_input}, {"output": ""})

        # Get the context (conversation history) from memory
        context = self.memory.load_memory_variables({})["chat_history"]

        # Define the mental health assistant prompt
        mental_health_prompt = (
            "You are an empathetic mental health assistant who listens and offers support. "
            "Your responses should be understanding, compassionate, and offer guidance when appropriate. "
            "If the user expresses feelings of grief, sadness, or any other challenging emotion, suggest resources or provide comfort. "
            "Avoid giving lengthy explanations but offer helpful suggestions and resources when the user asks for them."
            "Please do not include user input in the reponse"
            "Avoid repeating same sentences"
        )

        # Create the post-prompt including context
        post_prompt = (
            f"{mental_health_prompt}\n"
            f"Context: {context}\n"
            "Respond empathetically, offering emotional support, resources, or next steps when the user indicates interest."
        )

        # Build messages for the model
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": post_prompt},
        ]

        try:
            # Get response from Ollama using invoke
            response = self.llm.invoke(messages)

            # Extract and format the model response
            model_response = response.get('message', {}).get('content', '').strip() if not isinstance(response, str) else response.strip()
            print("Uncleaned response : ",model_response)    

            # Clean up the model response
            cleaned_response = clean_response(model_response)

            # Return the cleaned-up response
            return cleaned_response

        except Exception as e:
            raise Exception(f'Error contacting the model: {str(e)}')
