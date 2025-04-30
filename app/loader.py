import os
import asyncio
import json
import logging
import time
from uuid import uuid4
from tqdm import tqdm
from pdfminer.high_level import extract_text
from app.utils.splitter import TextSplitter
from app.openai import get_embeddings, token_size
from app.db import get_redis2, setup_db, add_chunks_to_vector_db
from app.config import settings
import google.generativeai as genai

logging.basicConfig(level=logging.ERROR)

def batchify(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]

async def process_docs(docs_dir=settings.DOCS_DIR):
    docs = []
    print('\nLoading documents')
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    for filename in tqdm(pdf_files):
        file_path = os.path.join(docs_dir, filename)
        text = extract_text(file_path)
        doc_name = os.path.splitext(filename)[0]
        docs.append((doc_name, text))
    print(f'Loaded {len(docs)} PDF documents')

    if not docs:
        print("No PDF documents found. Skipping PDF processing.")
        return []

    chunks = []
    text_splitter = TextSplitter(chunk_size=512, chunk_overlap=150)
    print('\nSplitting documents into chunks')
    for doc_name, doc_text in docs:
        doc_id = str(uuid4())[:8]
        doc_chunks = text_splitter.split(doc_text)
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunk = {
                'chunk_id': f'{doc_id}:{chunk_idx+1:04}',
                'text': chunk_text,
                'doc_name': doc_name,
                'vector': None
            }
            chunks.append(chunk)
        print(f'{doc_name}: {len(doc_chunks)} chunks')
    chunk_sizes = [token_size(c['text']) for c in chunks]
    print(f'\nTotal chunks: {len(chunks)}')
    print(f'Min chunk size: {min(chunk_sizes)} tokens')
    print(f'Max chunk size: {max(chunk_sizes)} tokens')
    print(f'Average chunk size: {round(sum(chunk_sizes)/len(chunks))} tokens')

    vectors = []
    print('\nEmbedding chunks')
    with tqdm(total=len(chunks)) as pbar:
        for batch in batchify(chunks, batch_size=64):
            batch_vectors = get_embeddings([chunk['text'] for chunk in batch])
            vectors.extend(batch_vectors)
            pbar.update(len(batch))

    for chunk, vector in zip(chunks, vectors):
        chunk['vector'] = vector
    return chunks

# def get_embeddings(inputs: list[str], model=settings.EMBEDDING_MODEL):
#     """
#     Mimics OpenAI batch embedding API.
#     """
#     embeddings = []
#     for i, inp in enumerate(inputs):
#         try:
#             data = {
#                 "model": "mxbai-embed-large",
#                 "prompt": inp,
#             }
#             response = requests.post(url='http://localhost:11434/api/embeddings', json=data)
#             response.raise_for_status()  # Raise an exception for bad status codes
#             data = response.json()
#             embedding = data.get("embeddings")
#             embeddings.append(embedding)
#             logger.debug(f"Embedding for item {i + 1} ('{inp}'): {embedding}")
#             print(f"Item {i + 1} done")
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error getting embedding for input '{inp}': {e}")
#             embeddings.append(None)  # Append None for failed embeddings
#         except KeyError:
#             logger.error(f"Error: 'embeddings' key not found in Ollama response for '{inp}'. Response: {data}")
#             embeddings.append(None)
#     return embeddings

async def process_json_dataset(dataset_dir=settings.DOCS_DIR):
    dataset_path = os.path.join(dataset_dir, "mini_dataset.json")
    logging.info(f"Checking for JSON dataset at: {dataset_path}") #added logging

    if not os.path.exists(dataset_path):
        logging.warning(f"JSON dataset not found at {dataset_path}. Skipping JSON processing.") #added logging
        return []

    print("\nProcessing JSON dataset...")
    try:
      with open(dataset_path, 'r') as f:
          data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading json data: {e}")
        return []

    if not data:
        logging.warning("JSON dataset is empty. Skipping JSON processing.") #added logging
        return []

    chunks = []
    for item in data:
        instruction = item["Instruction"]
        response = item["Response"]
        if not instruction or not response:
            logging.warning(f"Skipping JSON entry with empty instruction or response: {item}")
            continue

        text = f"Instruction: {instruction}\nResponse: {response}"
        doc_id = str(uuid4())[:8]
        chunk = {
            'chunk_id': f'{doc_id}:0001',
            'text': text,
            'doc_name': "dataset_json",
            'vector': None
        }
        chunks.append(chunk)

    chunk_sizes = [token_size(c['text']) for c in chunks]
    if not chunk_sizes:
        print("No valid JSON entries to process. Skipping JSON embedding.")
        return chunks

    print(f'\nTotal JSON entries: {len(chunks)}')
    print(f'Min entry size: {min(chunk_sizes)} tokens')
    print(f'Max entry size: {max(chunk_sizes)} tokens')
    print(f'Average entry size: {round(sum(chunk_sizes)/len(chunks))} tokens')

    vectors = []
    print('\nEmbedding JSON entries')
    with tqdm(total=len(chunks)) as pbar:
        for batch in batchify(chunks, batch_size=64):
            batch_vectors = get_embeddings([chunk['text'] for chunk in batch])
            vectors.extend(batch_vectors)
            pbar.update(len(batch))

    for chunk, vector in zip(chunks, vectors):
        chunk['vector'] = vector
    return chunks


async def load_knowledge_base():
    async with get_redis2() as rdb:
        print('Setting up Redis database')
        await setup_db(rdb)
        pdf_chunks = await process_docs()
        json_chunks = await process_json_dataset()
        all_chunks = pdf_chunks + json_chunks
        print('\nAdding chunks to vector db')
        await add_chunks_to_vector_db(rdb, all_chunks)
        print('\nKnowledge base loaded')

def main():
    asyncio.run(load_knowledge_base())

if __name__ == '__main__':
    main()
