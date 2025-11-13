# Import functions from individual modules
from DocumentLoader.loader import document_loader
from TextSpliter.spliter import text_spliting
from EmbeddingGeneration.generator import embedding_generation,store_embeddings_in_chromadb


def main():
    
    print("--- Starting Document Processing Pipeline ---")

    # Step 1: Load documents
    loaded_documents = document_loader()
    if not loaded_documents:
        print("No documents loaded. Pipeline halted.")
        return

    # Step 2: Split documents into chunks
    text_chunks = text_spliting(loaded_documents)
    if not text_chunks:
        print("No text chunks generated. Pipeline halted.")
        return

    # Step 3: Generate embeddings for chunks
    generated_embeddings = embedding_generation(text_chunks)
    if not generated_embeddings:
        print("No embeddings generated. Pipeline halted.")
        return

    # Step 4: Store embeddings and chunks in ChromaDB
    store_embeddings_in_chromadb(generated_embeddings, text_chunks)

    print("--- Document Processing Pipeline Completed ---")

# Entry point for the script
if __name__ == "__main__":
    main()
