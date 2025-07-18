import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

def create_code_vector_db(json_file_path: str, model_name: str, output_index_path: str, output_metadata_path: str):
    """
    Loads code chunks, filters them, generates embeddings, and saves a FAISS index
    along with corresponding metadata.

    Args:
        json_file_path (str): Path to the code_chunks.json file.
        model_name (str): The name of the SentenceTransformer model to use.
        output_index_path (str): Path to save the FAISS index file.
        output_metadata_path (str): Path to save the chunk metadata JSON file.
    """
    # 1. Load and Filter Chunks
    print(f"Loading chunks from '{json_file_path}'...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return

    # Filter for chunks that contain meaningful semantic information for a RAG system
    target_types = {'function', 'class', 'method', 'async_function', 'async_method'}
    filtered_chunks = [chunk for chunk in all_chunks if chunk.get('chunk_type') in target_types]
    
    if not filtered_chunks:
        print("No chunks of target types found. Exiting.")
        return
        
    print(f"Filtered chunks: Kept {len(filtered_chunks)} out of {len(all_chunks)} total chunks.")

    # 2. Prepare Text for Embedding
    # Combine code with metadata for richer semantic representation.
    texts_to_embed = []
    for chunk in filtered_chunks:
        # A good practice is to create a descriptive text for each chunk
        docstring = chunk.get('docstring', '') or "No docstring."
        name = chunk.get('name', '')
        chunk_type = chunk.get('chunk_type', '')
        
        # Create a descriptive header for the code content
        header = f"Type: {chunk_type}, Name: {name}\nDocstring: {docstring}\n---\n"
        prepared_text = header + chunk['content']
        texts_to_embed.append(prepared_text)

    # 3. Generate Embeddings
    print(f"Loading SentenceTransformer model: '{model_name}'...")
    # Using a model well-suited for code is beneficial, but a general one works too.
    # Consider models like 'microsoft/codebert-base' or 'all-MiniLM-L6-v2' for a start.
    model = SentenceTransformer(model_name).half()  # Convert the model to half precision for faster inference
    # model to fp16 for faster inference
    # model = SentenceTransformer(model_name, device='cpu').half()
    


    
    print("Generating embeddings for filtered chunks... (This may take a while)")
    # embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    # Define a batch size
    batch_size = 2 # You can adjust this number based on your VRAM

    print("Generating embeddings for filtered chunks... (This may take a while)")
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=batch_size, 
        show_progress_bar=True
    )

    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    print(f"Embeddings generated with dimension: {dimension}")

    # 4. Build and Save FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving FAISS index to '{output_index_path}'...")
    faiss.write_index(index, output_index_path)

    # 5. Save Metadata for Mapping
    # We need to save the original chunk info to map FAISS results back to the source code
    metadata_to_save = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "file_path": chunk.get("file_path"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "name": chunk.get("name"),
            "chunk_type": chunk.get("chunk_type")
        }
        for chunk in filtered_chunks
    ]
    
    print(f"Saving metadata mapping to '{output_metadata_path}'...")
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_to_save, f, indent=2)
        
    print("\nProcess complete!")
    print(f"FAISS index and metadata have been successfully saved.")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    CHUNKS_JSON_PATH = "code_chunks.json"
    
    # Recommended model for general purpose, good balance of speed and quality.
    # For more code-specific tasks, you might explore models like 'microsoft/codebert-base'.
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    
    OUTPUT_INDEX_PATH = "code_faiss.index"
    OUTPUT_METADATA_PATH = "code_metadata.json"
    
    # --- EXECUTION ---
    create_code_vector_db(
        json_file_path=CHUNKS_JSON_PATH,
        model_name=MODEL_NAME,
        output_index_path=OUTPUT_INDEX_PATH,
        output_metadata_path=OUTPUT_METADATA_PATH
    )

