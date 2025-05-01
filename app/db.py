import json
import numpy as np
import logging
from redis.asyncio import Redis
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.json.path import Path
from app.config import settings

VECTOR_IDX_NAME = 'idx:vector'
VECTOR_IDX_PREFIX = 'vector:'
CHAT_IDX_NAME = 'idx:chat'
CHAT_IDX_PREFIX = 'chat:'

def get_redis():
    return Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)

def get_redis2():
    return Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT2)


# VECTORS
async def create_vector_index(rdb):
    schema = (
        TextField('$.chunk_id', no_stem=True, as_name='chunk_id'),
        TextField('$.text', as_name='text'),
        TextField('$.doc_name', as_name='doc_name'),
        VectorField(
            '$.vector',
            'FLAT',
            {
                'TYPE': 'FLOAT32',
                'DIM': settings.EMBEDDING_DIMENSIONS,
                'DISTANCE_METRIC': 'COSINE'
            },
            as_name='vector'
        )
    )
    try:
        await rdb.ft(VECTOR_IDX_NAME).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=[VECTOR_IDX_PREFIX], index_type=IndexType.JSON)
        )
        print(f"Vector index '{VECTOR_IDX_NAME}' created successfully")
    except Exception as e:
        print(f"Error creating vector index '{VECTOR_IDX_NAME}': {e}")

VECTOR_IDX_PREFIX = "vector:"  # Set this to your actual prefix
BATCH_SIZE = 10  # Tune this value depending on load and network speed

async def add_chunks_to_vector_db(rdb, chunks):
    print(f'\nWriting {len(chunks)} chunks to Redis in batches of {BATCH_SIZE}')

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        pipe = rdb.pipeline(transaction=True)
        print(f"Batch {i+1} is uploading")
        for chunk in batch:
            try:
                key = VECTOR_IDX_PREFIX + chunk['chunk_id']
                pipe.json().set(key, Path.root_path(), chunk)
            except Exception as e:
                logging.error(f"Error preparing chunk {chunk['chunk_id']} for Redis: {e}")

        try:
            await pipe.execute()
        except Exception as e:
            logging.error(f"Error writing batch {i // BATCH_SIZE + 1} to Redis: {e}")

async def search_vector_db(rdb, query_vector, top_k=settings.VECTOR_SEARCH_TOP_K):
    query = (
        Query(f'(*)=>[KNN {top_k} @vector $query_vector AS score]')
        .sort_by('score')
        .return_fields('score', 'chunk_id', 'text', 'doc_name')
        .dialect(2)
    )
    
    console.log(query)
    
    res = await rdb.ft(VECTOR_IDX_NAME).search(query, {
        'query_vector': np.array(query_vector, dtype=np.float32).tobytes()
    })
    return [{
        'score': 1 - float(d.score),
        'chunk_id': d.chunk_id,
        'text': d.text,
        'doc_name': d.doc_name
    } for d in res.docs]

async def get_all_vectors(rdb):
    count = await rdb.ft(VECTOR_IDX_NAME).search(Query('*').paging(0, 0))
    res = await rdb.ft(VECTOR_IDX_NAME).search(Query('*').paging(0, count.total))
    return [json.loads(doc.json) for doc in res.docs]


# CHATS
async def create_chat_index(rdb):
    try:
        schema = (
            NumericField('$.created', as_name='created', sortable=True),
        )
        await rdb.ft(CHAT_IDX_NAME).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=[CHAT_IDX_PREFIX], index_type=IndexType.JSON)
        )
        print(f"Chat index '{CHAT_IDX_NAME}' created successfully")
    except Exception as e:
        print(f"Error creating chat index '{CHAT_IDX_NAME}': {e}")

async def create_chat(rdb, chat_id, created):
    chat = {'id': chat_id, 'created': created, 'messages': []}
    await rdb.json().set(CHAT_IDX_PREFIX + chat_id, Path.root_path(), chat)
    return chat

async def add_chat_messages(rdb, chat_id, messages):
    await rdb.json().arrappend(CHAT_IDX_PREFIX + chat_id, '$.messages', *messages)

async def chat_exists(rdb, chat_id):
    return await rdb.exists(CHAT_IDX_PREFIX + chat_id)

async def get_chat_messages(rdb, chat_id, last_n=None):
    if last_n is None:
        messages = await rdb.json().get(CHAT_IDX_PREFIX + chat_id, '$.messages[*]')
    else:
        messages = await rdb.json().get(CHAT_IDX_PREFIX + chat_id, f'$.messages[-{last_n}:]')

    if messages:
        formatted_messages = []
        for m in messages:
            if 'parts' in m:
                formatted_messages.append({'role': m['role'], 'parts': m['parts']})
            elif 'content' in m:
                formatted_messages.append({'role': m['role'], 'parts': m['content']})
            else:
                formatted_messages.append({'role': m['role'], 'parts': "No content provided"})
        return formatted_messages
    else:
        return []


async def get_chat(rdb, chat_id):
    return await rdb.json().get(chat_id)

async def get_all_chats(rdb):
    q = Query('*').sort_by('created', asc=False)
    count = await rdb.ft(CHAT_IDX_NAME).search(q.paging(0, 0))
    res = await rdb.ft(CHAT_IDX_NAME).search(q.paging(0, count.total))
    return [json.loads(doc.json) for doc in res.docs]


# GENERAL
async def setup_db(rdb):
    # Create the vector index (deleting the existing one if present)
    try:
        await rdb.ft(VECTOR_IDX_NAME).dropindex(delete_documents=True)
        print(f"Deleted vector index '{VECTOR_IDX_NAME}' and all associated documents")
    except Exception as e:
        pass
    finally:
        await create_vector_index(rdb)

    # Make sure that the chat index exists, and create it if it doesn't
    try:
        await rdb.ft(CHAT_IDX_NAME).info()
    except Exception:
        await create_chat_index(rdb)

async def clear_db(rdb):
    for index_name in [VECTOR_IDX_NAME, CHAT_IDX_NAME]:
        try:
            await rdb.ft(index_name).dropindex(delete_documents=True)
            print(f"Deleted index '{index_name}' and all associated documents")
        except Exception as e:
            print(f"Index '{index_name}': {e}")
