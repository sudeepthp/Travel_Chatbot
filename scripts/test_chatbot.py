from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_path = "../model/fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# def generate_response(prompt, max_length=100):
#     # Encode the input prompt
#     input_ids = tokenizer.encode(prompt, return_tensors='pt')

#     # Debug: Check input_ids shape and contents
#     print(f"Debug: Input IDs shape: {input_ids.shape}")
#     print(f"Debug: Input IDs: {input_ids}")

#     # Check if input_ids is empty
#     if input_ids.size()[1] == 0:
#         return "I'm sorry, I didn't understand that. Could you please rephrase?"

#     # Generate a response from the model
#     output = model.generate(
#         input_ids, 
#         max_length=max_length, 
#         num_beams=5, 
#         no_repeat_ngram_size=2, 
#         early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id
#     )

#     # Decode and return the response
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response

def generate_response(prompt, max_length=50):
    # Add context to steer the model
    prompt = "As a travel assistant, " + prompt

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Debug: Check input_ids shape and contents
    # print(f"Debug: Input IDs shape: {input_ids.shape}")
    # print(f"Debug: Input IDs: {input_ids}")

    # Check if input_ids is empty
    if input_ids.size()[1] == 0:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

    # Generate a response from the model
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=3, 
        no_repeat_ngram_size=2, 
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,  # Lower temperature for more focused responses
        top_p=0.9  # Use nucleus sampling for more controlled outputs
    )

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Test the chatbot with different inputs
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")
