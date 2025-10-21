from unsloth import FastModel
from datasets import load_from_disk, load_dataset, DatasetDict

SAVE_FOLDER_NAME = "gemma3_4b_it"

model, tokenizer = FastModel.from_pretrained(
    model_name = f"./{SAVE_FOLDER_NAME}/final-save-train", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit = True,
)

dataset = load_from_disk("hf_dataset")
test_dataset = dataset["test"]

correct_count = 0
for sample in test_dataset:
    prompt = f"Classify the following 5G fault description. Output only a single word: either 'network' or 'stress'. Do not provide any other text, explanations, or formatting.\n\nFault Description: {sample['text']}"
    messages = [{
        "role": "user",
        "content": [{"type" : "text", "text" : prompt,}]
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_tensors = "pt",
        return_dict = True,
    )
    outputs = model.generate(
        **inputs.to("cuda"),
        max_new_tokens = 64, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    format_resp = text_output[0].split("model")[1]
    print(f"The text prediction: {format_resp.strip()}")
    print(f"The actual output: {sample["label"]}")
    if format_resp.strip() == sample["label"].strip():
        print("CORRECT!!!!!")
        correct_count += 1
    print("\n\n")
print(f"Got {correct_count}/{len(test_dataset)} correct")
print(f"Accuracy: {correct_count/len(test_dataset)}")