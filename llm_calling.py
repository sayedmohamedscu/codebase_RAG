import retrive_docs


import json
from retrive_docs import load_faiss_index_and_metadata, retrieve_relevant_chunks, print_results

INDEX_PATH = "code_faiss.index"
METADATA_PATH = "code_metadata.json"
CHUNKS_JSON_PATH = "code_chunks.json"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Must match the model used in create_faiss.py
TOP_K = 5  # Number of results to retrieve

# --- EXECUTION ---
# Load FAISS index and metadata
index, metadata, chunks_dict = load_faiss_index_and_metadata(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
    chunks_json_path=CHUNKS_JSON_PATH
)

if index is None or metadata is None or chunks_dict is None:
    print("Failed to load index, metadata, or chunks. Exiting.")
    exit(1)

# Get user query
print("\nEnter your query (e.g., 'function to process text data'):")
# query = input("> ")
query= '''
Bug
when i add (cache=True)in Classification Training , the Ram using is increasing every epoch , until it crash the training , start like from 3 to 6 to 11 to 15 ....... 50 , GB
but if i don't add it , the ram using work fine , it be like 4 GB and all training is fixed

i work on colab
!yolo task=classify mode=train cache=True model=yolov8n-cls.pt data='/content/Classification-1' epochs=5  batch=265 imgsz=128 

Environment
No response

Minimal Reproducible Example
No response

Additional
No response'''
# Retrieve and display results
results = retrieve_relevant_chunks(
    query=query,
    model_name=MODEL_NAME,
    index=index,
    metadata=metadata,
    chunks_dict=chunks_dict,
    top_k=TOP_K
    )


print(print_results(results))
#call llm
# import requests
# import json
# import time
# import os

# sys_prompt = "You ar "
# # Set API key and API base for the custom API server
# api_key = os.getenv("API_KEY")  # Replace with your actual API key
# api_base_url = os.getenv("API_BASE_URL")  # Replace with your API base URL

# # Setup headers for the request
# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }

# # System message and query
# # sys_msg = "you are a helpful assistant"
# # query = "what is machine learning?"

# # Prepare the data payload for the POST request
# data = json.dumps({
#     "model": "Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
#     "messages": [
#         {"role": "system", "content":sys_prompt },
#         {"role": "user", "content": query}
#     ],
#     "temperature": 0.2
# })

# # Measure request execution time
# t1 = time.time()

# # Perform the POST request
# response = requests.post(f"{api_base_url}/chat/completions", headers=headers, data=data)
# print("Request time:", time.time() - t1)

# # Check the response and handle errors
# if response.status_code == 200:
#     # Parse response if request was successful
#     chat_response = response.json()
#     print("Chat response:", chat_response['choices'][0]['message']['content'])
# else:
#     # Print error information if something went wrong
#     print("Failed to fetch response:", response.status_code, response.text)

# print("this output based on this query :",query) 