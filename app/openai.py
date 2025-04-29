import google.generativeai as genai
from app.config import settings
import requests
import logging  # Import the logging module

# Initialize Gemini
genai.configure(api_key=settings.OPENAI_API_KEY)  # Assuming you still use this for something else

# Set model names
chat_model = genai.GenerativeModel(settings.MODEL)
# embedding_model = genai.GenerativeModel(settings.EMBEDDING_MODEL) # No longer needed

# Get a logger instance
logger = logging.getLogger(__name__)

# Approximate tokenizer since Gemini doesn't expose one
def token_size(text: str):
    # You can tweak this if you want a more accurate estimate
    return len(text.split())

def get_embedding(input: str, model=settings.EMBEDDING_MODEL):
    """
    Mimics OpenAI single embedding API.
    """
    try:
        data = {
            "model": "mxbai-embed-large",
            "prompt": input,  # Use the 'input' argument
        }
        response = requests.post(url='http://localhost:11434/api/embeddings', json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        embedding = data.get("embeddings")
        logger.debug(f"Embedding for '{input}': {embedding}")
        return embedding
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting embedding for '{input}': {e}")
        return None
    except KeyError:
        logger.error(f"Error: 'embeddings' key not found in Ollama response for '{input}'. Response: {data}")
        return None

def get_embeddings(inputs: list[str], model=settings.EMBEDDING_MODEL):
    """
    Mimics OpenAI batch embedding API.
    """
    embeddings = []
    for i, inp in enumerate(inputs):
        try:
            data = {
                "model": "mxbai-embed-large",
                "prompt": inp,
            }
            response = requests.post(url='http://localhost:11434/api/embeddings', json=data)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            embedding = data.get("embeddings")
            embeddings.append(embedding)
            logger.debug(f"Embedding for item {i + 1} ('{inp}'): {embedding}")
            print(f"Item {i + 1} done")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embedding for input '{inp}': {e}")
            embeddings.append(None)  # Append None for failed embeddings
        except KeyError:
            logger.error(f"Error: 'embeddings' key not found in Ollama response for '{inp}'. Response: {data}")
            embeddings.append(None)
    return embeddings

def chat_stream(messages: list[dict], model=settings.MODEL, temperature=0.1, **kwargs):
    """
    Mimics OpenAI chat stream API with Gemini.
    Accepts a list of dicts like: [{'role': 'user', 'content': 'Hi!'}, ...]
    """
    # Format prompt the Gemini way
    prompt = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
    )

    stream = chat_model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            **kwargs
        },
        stream=True
    )
    return stream

# Configure logging (optional, but helpful for debugging)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')