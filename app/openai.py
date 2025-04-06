import google.generativeai as genai
from app.config import settings

# Initialize Gemini
genai.configure(api_key=settings.OPENAI_API_KEY)

# Set model names
chat_model = genai.GenerativeModel(settings.MODEL)
# embedding_model = genai.GenerativeModel(settings.EMBEDDING_MODEL) # No longer needed

# Approximate tokenizer since Gemini doesn't expose one
def token_size(text: str):
    # You can tweak this if you want a more accurate estimate
    return len(text.split())

def get_embedding(input: str, model=settings.EMBEDDING_MODEL):
    """
    Mimics OpenAI single embedding API.
    """
    try:
        response = genai.embed_content( #corrected line
            model=model,
            content=input,
            task_type="retrieval_document"
        )
        return response.embedding.values
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_embeddings(inputs: list[str], model=settings.EMBEDDING_MODEL):
    """
    Mimics OpenAI batch embedding API.
    """
    embeddings = []
    for inp in inputs:
        try:
            response = genai.embed_content( #corrected line
                model=model,
                content=inp,
                task_type="retrieval_document"
            )
            embeddings.append(response.embedding.values)
        except Exception as e:
            print(f"Error getting embedding for input '{inp}': {e}")
            embeddings.append(None)  # Append None for failed embeddings
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