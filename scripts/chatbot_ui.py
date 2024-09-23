import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function to set up the chatbot model
def setup_chatbot():
    model_path = "../model/fine_tuned_model"  # Adjust the path as needed
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# Function to generate a response with improved context handling
def generate_response(prompt, history=[], max_length=50, max_context_length=2):
    tokenizer, model = st.session_state['chatbot']
    
    # Improved history formatting to avoid unnecessary repetition
    formatted_history = [
        f"User: {item['content']}\nAssistant: {history[idx + 1]['content']}" 
        for idx, item in enumerate(history[:-1]) if item['role'] == 'user' and history[idx + 1]['role'] == 'assistant'
    ]
    
    # Limit context length
    formatted_history = formatted_history[-max_context_length:]
    
    # Combine history with the current prompt, use refined format
    prompt_with_history = "\n".join(formatted_history) + f"\nUser: {prompt}\nAssistant:" if formatted_history else f"User: {prompt}\nAssistant:"
    
    # Truncate the prompt if it exceeds max_length to avoid errors
    if len(tokenizer.encode(prompt_with_history)) > max_length:
        prompt_with_history = prompt_with_history[-max_length:]
    
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt_with_history, return_tensors='pt')

    # Debug: Check input_ids shape and contents
    # st.write(f"Debug: Input IDs shape: {input_ids.shape}")
    # st.write(f"Debug: Input IDs: {input_ids}")

    # Generate a response from the model
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=3, 
        no_repeat_ngram_size=2, 
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.6,  # Lower temperature for more controlled and precise responses
        top_p=0.85,  # Use top_p sampling to focus on the most likely tokens
    )

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Strip the response to remove leading/trailing spaces and context prefixes
    response = response.replace(prompt_with_history, '').strip()
    return response

# Streamlit UI Setup
st.set_page_config(
    page_title="TravelBug",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Travel Chatbot 🤖")

# Load and setup chatbot if not already in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = setup_chatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input area
user_input = st.chat_input("Ask AI...")

# Process the user input and generate response
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display chatbot's response
    with st.chat_message("assistant"):
        response = generate_response(user_input, st.session_state.chat_history)
        assistant_response = response
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
