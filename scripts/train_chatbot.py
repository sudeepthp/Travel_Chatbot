from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the dataset in JSONL format
dataset = load_dataset('json', data_files={'train': '../data/dataset.jsonl'}, split='train')

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Load model and resize token embeddings if padding token is added
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Set max length to ensure consistency in padding
max_length = 128  # Adjust this value based on your dataset

# Function to flatten nested lists
def flatten_list(l):
    flat_list = []
    for item in l:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

# Tokenize the dataset with padding and truncation
def tokenize_function(example):
    # Tokenize query and response separately with same max length
    query_tokens = tokenizer(example["query"], truncation=True, padding='max_length', max_length=max_length)
    response_tokens = tokenizer(example["response"], truncation=True, padding='max_length', max_length=max_length)

    # Combine input_ids and attention_masks for both query and response
    input_ids = query_tokens['input_ids'] + response_tokens['input_ids']
    attention_mask = query_tokens['attention_mask'] + response_tokens['attention_mask']

    # Adjust labels to match the length of input_ids
    labels = response_tokens['input_ids'] + [tokenizer.pad_token_id] * (len(input_ids) - len(response_tokens['input_ids']))

    # Flatten all lists to avoid nested structures and verify data integrity
    input_ids = flatten_list(input_ids)
    attention_mask = flatten_list(attention_mask)
    labels = flatten_list(labels)

    # Handle None values and ensure all are lists of integers
    input_ids = input_ids if input_ids is not None else [tokenizer.pad_token_id] * max_length
    attention_mask = attention_mask if attention_mask is not None else [0] * max_length
    labels = labels if labels is not None else [tokenizer.pad_token_id] * max_length

    # Check data type consistency and convert to integers
    input_ids = list(map(int, input_ids))
    attention_mask = list(map(int, attention_mask))
    labels = list(map(int, labels))

    # Verify that each field is flat and has the same length
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
        print(f"Error: Length mismatch! input_ids: {len(input_ids)}, attention_mask: {len(attention_mask)}, labels: {len(labels)}")
        print(f"input_ids: {input_ids}\nattention_mask: {attention_mask}\nlabels: {labels}")
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # Return combined tokens ensuring everything is a list of integers
    return {
        'input_ids': input_ids,       # Ensure input_ids are flat list
        'attention_mask': attention_mask,  # Ensure attention_mask is flat list
        'labels': labels  # Ensure labels are flat list
    }

# Map the tokenize function over the dataset with debugging
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['query', 'response'])

# Split the dataset into train and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Define data collator with padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="../model/results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,  # Limit the number of saved checkpoints
)

# Initialize Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # Use data collator for padding
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('../model/fine_tuned_model')
tokenizer.save_pretrained('../model/fine_tuned_model')
