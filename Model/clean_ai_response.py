import re

def clean_response(response_text):
    # Define a more general regex pattern to capture various introductory phrases and roles
    intro_pattern = r"^(AI:?|Human:?|Assistant:?|Model:?|System:?|Here's a possible (AI|Assistant|Model|response):?|Response:?|Output:?|Reply:?|Result:?)\s*"
    
    # Use re.sub to remove any matched pattern at the beginning of the response
    cleaned_response = re.sub(intro_pattern, '', response_text, flags=re.IGNORECASE).strip()
    
    # Further cleaning to remove long, unnecessary phrases
    cleaned_response = re.sub(r"(I'll respond as AI would.*|Would you like.*|If you feel like.*|It's okay to.*|I'm here when you're ready.*|I'm here for you.*|I want to listen.*|without judgment.*)", "", cleaned_response).strip()

    # Ensure response is empathetic but concise; if empty, provide a fallback response
    cleaned_response = cleaned_response or "I'm here to listen. How are you feeling today?"

    return cleaned_response

# Example usage
# # uncleaned_response = "AI: Hello! I'm here to listen and support you. Is there something on your mind that you'd like to talk about?"
# cleaned_response = clean_response(uncleaned_response)
# print("Cleaned response:", cleaned_response)
