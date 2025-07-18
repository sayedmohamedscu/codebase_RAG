import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Loading SentenceTransformer model: '{model_name}'...")
device = 'cuda' if faiss.get_num_gpus() > 0 else 'cpu'
print(f"Using device: {device}")
model = SentenceTransformer(model_name, device=device)
if device == 'cuda':
    model = model.half()  # Use FP16 for GPU
def load_faiss_index_and_metadata(index_path: str, metadata_path: str, chunks_json_path: str) -> tuple:
    """
    Loads the FAISS index, metadata, and original code chunks.

    Args:
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata JSON file.
        chunks_json_path (str): Path to the original code_chunks.json file.

    Returns:
        tuple: (FAISS index, metadata list, chunks dictionary)
    """
    print(f"Loading FAISS index from '{index_path}'...")
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, None, None

    print(f"Loading metadata from '{metadata_path}'...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{metadata_path}' was not found.")
        return None, None, None

    print(f"Loading code chunks from '{chunks_json_path}'...")
    try:
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        # Create a dictionary for quick lookup by chunk_id
        chunks_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    except FileNotFoundError:
        print(f"Error: The file '{chunks_json_path}' was not found.")
        return None, None, None

    return index, metadata, chunks_dict

def retrieve_relevant_chunks(query: str, model_name: str, index: faiss.IndexFlatL2, metadata: List[Dict], chunks_dict: Dict, top_k: int = 5) -> List[Dict]:
    """
    Encodes the query and retrieves the top-k most relevant code chunks.

    Args:
        query (str): The user's input query.
        model_name (str): The SentenceTransformer model to use.
        index (faiss.IndexFlatL2): The loaded FAISS index.
        metadata (List[Dict]): The metadata for the indexed chunks.
        chunks_dict (Dict): Dictionary mapping chunk_id to chunk details.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict]: List of dictionaries containing the retrieved chunks and their metadata.
    """
    # Load the model
    

    # Prepare query text (mimic the chunk format used during indexing)
    query_text = f"Type: query\nDocstring: {query}\n---\n{query}"
    query_embedding = model.encode([query_text], show_progress_bar=False).astype('float32')

    # Perform FAISS search
    print(f"Searching for top {top_k} relevant chunks...")
    distances, indices = index.search(query_embedding, top_k)

    # Collect results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(metadata):
            meta = metadata[idx]
            chunk_id = meta['chunk_id']
            chunk = chunks_dict.get(chunk_id, {})
            results.append({
                'chunk_id': chunk_id,
                'file_path': meta.get('file_path', 'Unknown'),
                'start_line': meta.get('start_line', -1),
                'end_line': meta.get('end_line', -1),
                'name': meta.get('name', 'Unknown'),
                'chunk_type': meta.get('chunk_type', 'Unknown'),
                'docstring': chunk.get('docstring', 'No docstring.'),
                'content': chunk.get('content', 'No content available.'),
                'distance': float(distance)  # Similarity score (L2 distance)
            })
        else:
            print(f"Warning: Index {idx} out of range for metadata.")

    return results

# def print_results(results: List[Dict]):
#     """
#     Prints the retrieved results in a readable format.

#     Args:
#         results (List[Dict]): List of retrieved chunk details.
#     """
#     if not results:
#         print("No relevant chunks found.")
#         return

#     print("\n=== Retrieved Chunks ===")
#     returned_text=""
#     for i, result in enumerate(results, 1):
#         # print(f"\nResult {i}:")
#         # print(f"Chunk ID: {result['chunk_id']}")
#         # print(f"Type: {result['chunk_type']}")
#         # print(f"Name: {result['name']}")
#         # print(f"File: {result['file_path']} (Lines {result['start_line']}–{result['end_line']})")
#         # print(f"Distance: {result['distance']:.4f}")
#         # print(f"Docstring: {result['docstring']}")
#         # print("\nCode:")
#         # print(result['content'])
#         # print("-" * 80)
#         returned_text=returned_text + "\n" +"chunk_id: " + "\n"+ f"File: {result['file_path']} (Lines {result['start_line']}–{result['end_line']})" + "\n" + result['chunk_id'] + "\n" +"code: " + result['content'] 
#     # return in style 
#     return returned_text
#     #return { {'results': for result in results  }
# # In retrive_docs.py

def print_results(results: List[Dict]):
    """
    Formats the retrieved results into a Markdown string with GitHub links
    and syntax highlighting.

    Args:
        results (List[Dict]): List of retrieved chunk details.
    """
    if not results:
        return "No relevant chunks found."

    GITHUB_BASE_URL = "https://github.com/ultralytics/ultralytics/blob/main/"
    markdown_output = ""

    for i, result in enumerate(results, 1):
        file_path = result.get('file_path', 'Unknown')
        start_line = result.get('start_line', -1)
        end_line = result.get('end_line', -1)

        # Construct a direct link to the code on GitHub
        if file_path != 'Unknown' and start_line != -1:
            github_link = f"{GITHUB_BASE_URL}{file_path}#L{start_line}-L{end_line}"
            header = f"### {i}. [{file_path}]({github_link}) (Lines {start_line}–{end_line})"
        else:
            header = f"### {i}. {file_path} (Lines {start_line}–{end_line})"

        markdown_output += f"{header}\n"
        markdown_output += f"**Type:** `{result.get('chunk_type', 'N/A')}`  **Name:** `{result.get('name', 'N/A')}`\n\n"
        markdown_output += "```python\n"
        markdown_output += result.get('content', '# No content available.') + "\n"
        markdown_output += "```\n---\n"

    return markdown_output

# if __name__ == "__main__":
#     # --- CONFIGURATION ---
#     INDEX_PATH = "code_faiss.index"
#     METADATA_PATH = "code_metadata.json"
#     CHUNKS_JSON_PATH = "code_chunks.json"
#     MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Must match the model used in create_faiss.py
#     TOP_K = 5  # Number of results to retrieve

#     # --- EXECUTION ---
#     # Load FAISS index and metadata
#     index, metadata, chunks_dict = load_faiss_index_and_metadata(
#         index_path=INDEX_PATH,
#         metadata_path=METADATA_PATH,
#         chunks_json_path=CHUNKS_JSON_PATH
#     )

#     if index is None or metadata is None or chunks_dict is None:
#         print("Failed to load index, metadata, or chunks. Exiting.")
#         exit(1)

#     # Get user query
#     print("\nEnter your query (e.g., 'function to process text data'):")
#     query = input("> ")

#     # Retrieve and display results
#     results = retrieve_relevant_chunks(
#         query=query,
#         model_name=MODEL_NAME,
#         index=index,
#         metadata=metadata,
#         chunks_dict=chunks_dict,
#         top_k=TOP_K
#     )

#     print_results(results)