import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

# Test queries
test_queries = [
    "Hello!", 
    "Can you help me book a flight to New York?", 
    "What are the top attractions in Rome?",
    "What are the best beaches in Bali?", 
    "I need a hotel in Paris."
]

# Generate responses for the test queries
for query in test_queries:
    print(f"Query: {query}")
    response = generate_response(query)
    print(f"Response: {response}\n")

