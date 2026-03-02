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


max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
RUN_NAME = "log_faults_precomputed_orig"
SAVE_ROOT_DIRECTORY = "trained_models/dirty_datasetv1"
SAVE_FOLDER_NAME = f"{SAVE_ROOT_DIRECTORY}/{RUN_NAME}"
TRAIN_DATASET_PATH = "./datasets/combined/dirty_datasetv1.csv"

train_dataset = load_dataset("csv", data_files=TRAIN_DATASET_PATH, split="train")
print(f"Sample: {train_dataset[0]['question']}, {train_dataset[0]['answer']}")

print(train_dataset.column_names)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,  # Choose any for long context!
    load_in_4bit=False,  # 4 bit quantization to reduce memory
    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
)
print(model)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32 * 2,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=True,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


def transform_conversation(sample):
    return {
        "conversations": [
            {"from": "human", "value": sample["question"]},
            {"from": "gpt", "value": sample["answer"]},
        ]
    }


train_dataset = train_dataset.map(transform_conversation)
train_dataset = standardize_data_formats(train_dataset)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {
        "text": texts,
    }


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

print(train_dataset[0]["text"])


# early_stopping_callback = EarlyStoppingCallback(
#     early_stopping_patience=3, # Stop after 3 evaluations with no improvement
#     early_stopping_threshold=0.01 # A small threshold to prevent stopping on minor fluctuations
# )

grad_acc_steps = 1
train_batch_size = 8
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
        per_device_train_batch_size=train_batch_size,
        # basically using batch size of 2*4=8
        gradient_accumulation_steps=grad_acc_steps,
        # around 10% of steps/epoch. increases lr gradually
        warmup_steps=math.ceil(steps_per_epoch_ceil * 0.1),
        num_train_epochs=1,
        # max_steps=30, # default is -1
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,  # use low numbers. It penalizes large weights to prevent overfitting
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="tensorboard",
        logging_dir=os.path.join(SAVE_ROOT_DIRECTORY, "logs", RUN_NAME),
        run_name=RUN_NAME,
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
        per_device_eval_batch_size=2,  # batch size for evaluation
        load_best_model_at_end=True,  # save best model based on eval loss
        metric_for_best_model="eval_loss",
        max_grad_norm=0.5,
        gradient_checkpointing="unsloth",  # "unsloth" for reduced memory
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

print(
    tokenizer.decode(
        [
            tokenizer.pad_token_id if x == -100 else x
            for x in trainer.train_dataset[0]["labels"]
        ]
    ).replace(tokenizer.pad_token, " ")
)

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
