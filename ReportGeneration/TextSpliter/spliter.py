from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document type for clarity

def text_spliting(docs: list[Document]) -> list[str]:
    """
    Splits a list of Langchain Document objects into smaller text chunks.

    Args:
        docs (list[Document]): A list of Langchain Document objects, each with a 'page_content' attribute.

    Returns:
        list[str]: A list of strings, where each string is a text chunk.
    """
    if not docs:
        print("No documents provided for splitting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=512,
    )

    # Extract page_content from each Document object and join them into a single string
    # before passing to split_text. Use a suitable separator like double newline.
    all_text = "\n\n".join([doc.page_content for doc in docs])

    # Perform the split on the combined text
    chunks = splitter.split_text(all_text)

    return chunks