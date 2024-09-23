import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model and tokenizer
model_path = '../model/fine_tuned_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the padding token is added, if needed
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Define a function to generate a response from a query
def generate_response(query, max_length=128, temperature=0.7, top_k=50, top_p=0.95):
    # Tokenize the input query
    input_ids = tokenizer.encode(query, return_tensors='pt')

    # Generate the response using the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,  # Adjust temperature to control randomness
            top_k=top_k,  # Consider only the top-k logits for next token
            top_p=top_p,  # Consider tokens with cumulative probability p
            do_sample=True,  # Enables stochastic sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output into text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit App UI
st.title("Travel Assistant AI")

# Input box for the user's query
user_query = st.text_input("Enter your travel-related query:", "")

# Button to trigger the response generation
if st.button("Generate Response"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        # Generate the response
        response = generate_response(user_query)
        st.write(f"**Response:** {response}")

# Add a sidebar with model configuration
st.sidebar.title("Settings")
max_length = st.sidebar.slider("Max Length", min_value=50, max_value=300, value=128, step=10)
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7)
top_k = st.sidebar.slider("Top-K", min_value=10, max_value=100, value=50)
top_p = st.sidebar.slider("Top-P", min_value=0.5, max_value=1.0, value=0.95)

# Display an about section
st.sidebar.title("About")
st.sidebar.info(
    """
    This is an AI-powered travel assistant demo using a fine-tuned GPT-2 model.
    You can ask questions about travel plans, destinations, hotels, and more!
    """
)
