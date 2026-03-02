from datasets import load_from_disk, load_dataset, DatasetDict
from collections import Counter
from unsloth import FastModel, FastLanguageModel
import torch
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
from unsloth.chat_templates import get_chat_template
from transformers import EarlyStoppingCallback
import math
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
import os


max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
SAVE_FOLDER_NAME = "qwen3_8b"

dataset = load_from_disk("hf_dataset")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_labels = [example['label'] for example in train_dataset]
label_counts = Counter(train_labels)
print(f"Train Label counts: {label_counts}\n")


test_labels = [example['label'] for example in test_dataset]
label_counts = Counter(test_labels)
print(f"Test Label counts: {label_counts}\n")

print(f"Sample of dataset: {train_dataset[100]['text']}, {train_dataset[100]['label']}")

print(train_dataset.column_names, test_dataset.column_names)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def transform_conversation(sample):
    return {
        'conversations': [
            {
                'from': 'human',
                'value': f"Classify the following 5G fault description. Output only a single word: either 'network' or 'stress'. Do not provide any other text, explanations, or formatting.\n\nFault Description: {sample['text']}"
            },
            {
                 'from': 'gpt',
                'value': sample['label']
            }
        ]
    }
train_dataset = train_dataset.map(transform_conversation)
train_dataset = standardize_data_formats(train_dataset)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-3",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>") for convo in convos]
    return {"text": texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
train_eval_split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_eval_dataset = DatasetDict(
    {
        "train": train_eval_split["train"],
        "validation": train_eval_split["test"],
    }
)
train_dataset = train_eval_dataset["train"]
eval_dataset = train_eval_dataset["validation"]
print(len(train_dataset), len(eval_dataset))

print(train_dataset[0]['text'])


# early_stopping_callback = EarlyStoppingCallback(
#     early_stopping_patience=3, # Stop after 3 evaluations with no improvement
#     early_stopping_threshold=0.01 # A small threshold to prevent stopping on minor fluctuations
# )

grad_acc_steps = 4
train_batch_size = 2
steps_per_epoch = len(train_dataset) / (grad_acc_steps * train_batch_size)
steps_per_epoch_ceil = math.ceil(steps_per_epoch)
print(steps_per_epoch_ceil, " is the steps/epoch\n\n")
log_save_eval_steps = max(1, steps_per_epoch_ceil // 4)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # callbacks=[early_stopping_callback],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # basically using batch size of 2*4=8
        warmup_steps=math.ceil(steps_per_epoch_ceil * 0.1),  # around 10% of steps/epoch. increases lr gradually
        num_train_epochs=3,
        # max_steps=30, # default is -1
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01, # use low numbers. It penalizes large weights to prevent overfitting
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="tensorboard",
        logging_dir=f"./{SAVE_FOLDER_NAME}/checkpoint/logs",
        logging_steps=log_save_eval_steps,

        dataset_num_proc=2,
        save_strategy="steps",
        save_steps=log_save_eval_steps,
        save_total_limit=3,
        greater_is_better=False,
        output_dir=f"./{SAVE_FOLDER_NAME}/checkpoint",
        # evaluation configs
        eval_strategy="steps",
        eval_steps=log_save_eval_steps,
        per_device_eval_batch_size=2,    # batch size for evaluation
        load_best_model_at_end=True,     # save best model based on eval loss
        metric_for_best_model="eval_loss",

        max_grad_norm=0.5,
        gradient_checkpointing="unsloth", # "unsloth" for reduced memory
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
print("TESTING:")
print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))

if os.path.isdir(trainer.args.output_dir):
    last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
else:
    trainer.train()

model.save_pretrained(f"./{SAVE_FOLDER_NAME}/final-save-train")  # Local saving
tokenizer.save_pretrained(f"./{SAVE_FOLDER_NAME}/final-save-train")
