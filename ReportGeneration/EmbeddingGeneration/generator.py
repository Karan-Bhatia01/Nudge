import os
import chromadb
from dotenv import load_dotenv
from google import genai  # ✅ Using Google’s official genai client

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)

def embedding_generation(chunks):
    """
    Generates embeddings for a list of text chunks using Google Gemini Embedding API.
    
    Args:
        chunks (list[str]): List of text chunks.

    Returns:
        list[list[float]]: List of embedding vectors.
    """
    if not chunks:
        print("⚠️ No chunks provided for embedding.")
        return None

    embeddings = []
    for chunk in chunks:
        try:
            result = client.models.embed_content(
                model="models/embedding-001",
                contents=chunk
            )
            if result and hasattr(result, "embeddings"):
                embeddings.append(result.embeddings[0].values)
            else:
                print("⚠️ Unexpected embedding response format.")
                embeddings.append([0.0])
        except Exception as e:
            print(f"⚠️ Error embedding chunk: {e}")
            embeddings.append([0.0])

    return embeddings


def store_embeddings_in_chromadb(
    embeddings, chunks,
    collection_name="my_document_embeddings",
    db_path="./chroma_db"
):
    """
    Stores text chunks and their embeddings in a ChromaDB collection.

    Args:
        embeddings (list[list[float]]): List of embedding vectors.
        chunks (list[str]): List of corresponding text chunks.
        collection_name (str): ChromaDB collection name.
        db_path (str): Directory path to store ChromaDB data.
    """
    if not embeddings or not chunks:
        print("⚠️ No embeddings or chunks to store.")
        return

    if len(embeddings) != len(chunks):
        raise ValueError("❌ Number of embeddings must match number of chunks.")

    try:
        # Initialize persistent ChromaDB client
        client = chromadb.PersistentClient(path=db_path)

        # Get or create collection
        collection = client.get_or_create_collection(name=collection_name)

        # Generate unique IDs for each chunk
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids
        )

        print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'.")
        print(f"📂 Database path: {db_path}")

    except Exception as e:
        print(f"❌ Error storing embeddings in ChromaDB: {e}")
