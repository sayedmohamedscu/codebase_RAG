# app.py
import gradio as gr
import requests
import json
import os
from retrive_docs import load_faiss_index_and_metadata, retrieve_relevant_chunks ,print_results

# --- CONFIGURATION ---
INDEX_PATH = "code_faiss.index"
METADATA_PATH = "code_metadata.json"
CHUNKS_JSON_PATH = "code_chunks.json"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 10  # Number of results to retrieve for context

# --- SYSTEM PROMPT ---
# This prompt is crucial for guiding the LLM's behavior.
SYSTEM_PROMPT = """
You are an expert software developer and technical analyst. Your task is to help a user understand a codebase and debug potential issues.

You have been provided with a user's question and a set of the most relevant code chunks retrieved from the codebase based on their query.

Your mission is to synthesize this information and provide a clear, accurate, and helpful response.

Follow these instructions carefully:
1.  **Analyze the Goal:** First, understand the user's primary goal. Are they reporting a bug, asking for an explanation, or trying to understand how something works?
2.  **Base Your Answer on Provided Context:** Your primary source of truth is the retrieved code chunks. Ground your entire analysis in the code provided. Do not invent functionality or assume the existence of code that is not present in the context.
3.  **Directly Address the Query:** Directly answer the user's question. If the context contains a definitive answer (e.g., a warning message about a known bug), state it clearly and quote the relevant code.
4.  **Synthesize and Hypothesize:** If the answer is not immediately obvious, synthesize information from multiple chunks. Form a hypothesis about the cause of the bug or the functionality in question, explaining your reasoning by referencing specific lines of code.
5.  **Provide Actionable Recommendations:** Conclude with clear, actionable advice. This could be a suggested code change, a command to run, or a recommendation to avoid a specific feature based on the evidence in the code.
6.  **Acknowledge Limitations:** If the provided code chunks are insufficient to fully answer the question, state this clearly. Explain what additional information would be needed.
7.  **Structure Your Response:** Format your response using Markdown for readability. Use code blocks for code snippets and bold text to highlight key findings.
8. **show output reference at the end:** to keep a trust show the source where you get the information from, like if it included in line .. or code ...  if available only in the context. 
"""

# --- LOAD DATA ON STARTUP ---
print("--- Initializing Application ---")
# Check if all required files exist before launching the UI
if not all(os.path.exists(p) for p in [INDEX_PATH, METADATA_PATH, CHUNKS_JSON_PATH]):
    print("ERROR: One or more required data files are missing.")
    print("Please make sure 'code_faiss.index', 'code_metadata.json', and 'code_chunks.json' are in the same directory.")
    # Gradio doesn't have a clean way to exit, so we'll show an error in the UI
    index, metadata, chunks_dict = None, None, None
else:
    index, metadata, chunks_dict = load_faiss_index_and_metadata(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        chunks_json_path=CHUNKS_JSON_PATH
    )
print("--- Initialization Complete ---")


def get_expert_analysis(api_key, api_url, llm_model_name, user_query):
    """
    The main function that orchestrates the RAG pipeline.
    """
    if not all([api_key, api_url, llm_model_name, user_query]):
        return "Error: API Key, API URL, Model Name, and Question are all required."

    if index is None:
        return "Error: FAISS index and data could not be loaded. Please check the console for errors and restart."

    # 1. RETRIEVAL: Get relevant code chunks
    print("\n--- Starting Retrieval ---")
    retrieved_results = retrieve_relevant_chunks(
        query=user_query,
        model_name=EMBEDDING_MODEL_NAME,
        index=index,
        metadata=metadata,
        chunks_dict=chunks_dict,
        top_k=TOP_K
    )

    if not retrieved_results:
        return "Could not find any relevant code chunks for your query. Please try rephrasing it."

    context_str = print_results(retrieved_results)
    
    print("--- Starting Generation ---")
    final_user_prompt = f"""
{context_str}

--- User's Question ---
{user_query}

--- Analysis and Answer ---
Based on the provided code context, here is the analysis of your question:
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_user_prompt}
        ]
    }

    try:
        print(f"Sending request to LLM: {llm_model_name} at {api_url}")
        # Use the provided api_url from the Gradio input
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        response_json = response.json()
        llm_answer = response_json['choices'][0]['message']['content']
        print("--- Generation Complete ---")
        
        # Correctly format the final response to render Markdown
        full_response = f"## Expert Analysis\n\n{llm_answer}\n\n---\n\n### Retrieved Context\n\nThis analysis was based on the following retrieved code chunks:\n\n{context_str}"
        return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return f"Error: Failed to connect to the LLM API. Please check your API URL, API key, and network connection.\n\nDetails: {e}"
    except (KeyError, IndexError) as e:
        print(f"Error parsing LLM response: {e}")
        return f"Error: Received an unexpected response from the LLM API. Please check the model name and try again.\n\nResponse: {response.text}"


# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Code Assistant") as demo:
    gr.Markdown("# RAG-Powered Code Assistant")
    gr.Markdown("This tool uses a local code database (FAISS) and a Large Language Model (LLM) to answer questions about your codebase.")

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="Enter your API key here"
            )
            # New input field for the API URL
            api_url_input = gr.Textbox(
                label="API Endpoint URL",
                value="https://openrouter.ai/api/v1/chat/completions",
                placeholder="Enter the chat completions endpoint URL"
            )
            llm_model_input = gr.Dropdown(
                label="Select LLM Model",
                choices=[
                    "moonshotai/kimi-k2:free",
                    "mistralai/devstral-small-2505:free",
                    "qwen/qwen3-235b-a22b:free",
                    "deepseek/deepseek-chat-v3-0324:free",
                ],
                value="moonshotai/kimi-k2:free"
            )
            user_query_input = gr.Textbox(
                label="Your Question / Bug Report",
                lines=8,
                placeholder="e.g., 'When I use cache=True, my RAM usage explodes. Why?'"
            )
            submit_button = gr.Button("Get Analysis", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## Analysis Result")
            output_text = gr.Markdown()

    # Update the inputs list for the click event
    submit_button.click(
        fn=get_expert_analysis,
        inputs=[api_key_input, api_url_input, llm_model_input, user_query_input],
        outputs=output_text
    )
    
    gr.Examples(
        examples=[
            [
                "When I use cache=True in classification training, the RAM usage increases with every epoch and crashes. Why?",
            ],
            [
                "How does the autobatch function work?",
            ]
        ],
        inputs=user_query_input
    )


if __name__ == "__main__":
    demo.launch(share=True)
