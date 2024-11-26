import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
from pinecone import Pinecone
from datetime import datetime
import pandas as pd
import json

# # Configration Loader function
# def load_config(config_path="config.json"):
#     """
#     Load configuration from a JSON file.

#     Args:
#         config_path (str): Path to the configuration JSON file.
    
#     Returns:
#         dict: Dictionary containing the configuration.
#     """
#     try:
#         with open(config_path, "r") as file:
#             config = json.load(file)
#         return config
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Configuration file not found at {config_path}")
#     except json.JSONDecodeError:
#         raise ValueError("Invalid JSON in the configuration file")

# # Load configuration
# config = load_config("config.json")

# Initialize Pinecone and Google Gemini with API keys
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
index = pc.Index("intelligent-search-v2")

# Set up OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

print("Setting up model configurations...")  # Debug print

# Initialize SentenceTransformer model for query embedding
model = SentenceTransformer(st.secrets["sentence_transformer"]["model"])

# Streamlit UI setup
st.title("Chatbot with Pinecone and GPT")

# User input field for query
query = st.text_input("Enter your query:")

# Check if a query has been entered
if query:
    print(f"User query: {query}")  # Debug print
    model_used = "None"  # Variable to track which model was used

    # Generate embedding for the query using SentenceTransformer
    embedding1 = model.encode(query)
    print(f"Query embedding: {embedding1[:5]}... (truncated)")  # Debug print to show a sample of the embedding

    # Perform a search in Pinecone using the query embedding
    try:
        answer = index.query(
            namespace="",              # Search within the default namespace
            vector=embedding1.tolist(),  # Convert embedding to a list format
            top_k=5,                    # Retrieve the top 3 matches
            include_metadata=True       # Include metadata for matched vectors
        )
        print(f"Pinecone query response: {answer}")  # Debug print
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        print(f"Error querying Pinecone: {e}")  # Debug print
        answer = {"matches": []}  # Default to an empty response on error

    # Extract relevant texts from search results
    retrieved_texts = [match['metadata']['text'] for match in answer['matches']]
    retrieved_pdf_title = [match['metadata']['title'] for match in answer['matches']]
    retrieved_pdf_page = [match['metadata']['page_number'] for match in answer['matches']]
    retrieved_pdf_link = [match['metadata']['link'] for match in answer['matches']]

    if retrieved_texts:
        print("Retrieved texts from Pinecone:")  # Debug print
        for text in retrieved_texts:
            print(f"- {text}")  # Debug print
   
        # Determine temperature and max_tokens based on query length or complexity
        max_tokens = 200 if len(query) < 50 else 300  # More tokens for complex questions

        # Refine prompt structure dynamically
        prompt = (
            f"Context: {' '.join(retrieved_texts)}\n\n"
            f"User Query: {query}\n\n"
            "Task:\n"
            "- Analyze the provided Context to address the User Query effectively."
            "- Use bullet points for clarity if multiple aspects are present."
            "- Ensure if need then combine all given context to answer the User query."
            "- If the Context provides indirect or scattered details, synthesize them to form a coherent answer."
            "- If the Context does not explicitly mention the answer, provide reasonable inferences based on the available details."
            "- If the Context contains no relevant information, respond with: 'Context does not provide sufficient information to answer the question.'\n\n"
            "- Ensure that sentences with the same meaning are not repeated in the response."
            "Note: Ensure that the response is **strictly based on the provided Context** and avoid introducing external information."
        )

        try:
                    gpt4_response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an intelligent assistant that provides answers based on given context."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=st.secrets["openai"]["temperature"]
                    )
                    response_text = gpt4_response['choices'][0]['message']['content']
                    gpt4_token_usage = gpt4_response['usage']['total_tokens']
                    model_used = "GPT-4"  # Update model used

                    # Log token usage to file
                    with open("gpt_token_log.txt", "a") as log_file:
                        log_file.write(f"{datetime.now()} | Model: GPT-4 | Tokens Used: {gpt4_token_usage}\n")

                    print(f"GPT-4 response: {response_text}")  # Debug print
                    print(f"GPT-4 token usage: {gpt4_token_usage}")  # Debug print
                    
        except Exception as e:
                    st.error(f"Error generating response with GPT-4: {e}")
                    print(f"Error generating response with GPT-4: {e}")  # Debug print
                    response_text = "No response generated."

        # Display the response in the Streamlit app
        if response_text:
                st.write(response_text)
                # Feedback code
                st.write("Was this response helpful?")
                sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
                selected = st.feedback("thumbs")
                if selected is not None:
                    with open(st.secrets["feedback_log_file"], "a") as feedback_log:
                        feedback_log.write(f"{datetime.now()} | Query: {query} | Feedback: {selected} | Model used: {model_used}\n")
        else:
            st.write("No response generated.")

        #Display the retrieved texts for reference
        st.write("YOU CAN REFER TO THESE DOCUMENTS:")
        df=pd.DataFrame()
        df['Pdf']=retrieved_pdf_title
        df['Page']=retrieved_pdf_page
        df['Link']=retrieved_pdf_link
        st.dataframe(
            df,
            column_config={
                "Pdf": "Pdf",
                "Page": "Page",
                "Link": st.column_config.LinkColumn("Link"),
                
            },
            hide_index=True,
        )
        st.write(retrieved_texts)         
    else:
        print("No relevant texts found.")  # Debug print
        st.write("Pincone have no relevent context")

