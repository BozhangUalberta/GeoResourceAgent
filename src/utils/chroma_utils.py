import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from decouple import config 
from functools import wraps


# Global client
chroma_client = None
collection_list = {}

async def init_chroma_client():
    global chroma_client
    if chroma_client is None:
        chroma_client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
        print("Chroma client initialized.")


# Define a decorator
def chroma(func):
    """
    A Helper decorator for chroma related operations.
    It automatically takes first arg: "userID" and return a arg: "collection" (recommand to put it at the end of all args)
    
    function should be defined like:
    @chroma
    async def func_name(userID, arg1, arg2, ..., collection):
        code 1
        code 2
        return something
    
    "collection" is specifically registered for each userID, it is the third level in chroma definition (see Chorma cookbook for details).
    "collection" is operated by a async chroma client which registered to chromaDB with FastAPI lifespan.
    Use "collection" for all related operations rather get it manually.
    """
    @wraps(func)
    async def wrapper(userID, *args, **kwargs):
        global chroma_client
        global collection_list

        if userID not in collection_list:
            emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=config("OPENAI_API_KEY"),
                        model_name="text-embedding-3-small"
                    )

            collection = await chroma_client.get_or_create_collection(name=userID, embedding_function=emb_fn)
            collection_list[userID] = collection

        kwargs['collection'] = collection_list[userID]
        # Call the original function
        return await func(userID, *args, **kwargs)
    return wrapper


@chroma
async def add_docs(userID, documents, metadatas, collection):
    """
    Add documents into chroma collection with "userID".

    return:
        1 - if all succeed, else will be catched by exceptions.
    """
    current_len = await collection.count()

    await collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=[f"id{x}" for x in range(current_len, current_len+len(metadatas))],
    )

    return 1

@chroma
async def query_docs(userID, query, collection):
    """
    Get most relevant several documents from chroma collection "userID".

    return:
        dict:
            {
                "ids": [id1, id2, id3],
                "distance": [num1, num2, num3],
                "metadata": ...,
                "documents": ...,
            }
    """
    results = await collection.query(
        query_texts=[query],
        n_results=5,
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains":"search_string"}
    )

    return results

