from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_pdf(folder_path):
    """
    Loads all PDF files from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        list: A list of documents loaded from all PDFs.
    """
    
    documents = []
    full_text = ""
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                for doc in documents:
                    full_text += doc.page_content
        return documents, full_text
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF file: {e}")



def chunk_text(documents, full_text, CHUNK_SIZE, CHUNK_OVERLAP):
    """
    Split the text into chunks and assign metadata to each chunk.
    
    Args:
        documents (list): A list of documents.
        full_text (str): The full text to be split into chunks.
    
    Returns:
        list: A list of chunks with metadata.
    """
    
    # Split the text into chunks and assign metadata to each chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    raw_chunks = text_splitter.split_text(full_text)

    # Assign metadata to chunks
    chunks = []
    for doc in documents:
        current_page_text = doc.page_content
        current_page_metadata = doc.metadata
        for chunk in raw_chunks:
            if chunk in current_page_text or chunk[:50] in current_page_text:
                if not chunk in current_page_text:
                    chunks.append(
                        {
                            "text": chunk.strip(),
                            "metadata": {
                                "source":current_page_metadata["source"],
                                "page":current_page_metadata["page_label"] + "," + str(int(current_page_metadata["page_label"]) + 1)
                            }
                        }
                    )
                    break
                chunks.append(
                        {
                            "text": chunk.strip(),
                            "metadata": {
                                "source":current_page_metadata["source"],
                                "page":current_page_metadata["page_label"]
                            }
                        }
                    )
    return chunks
