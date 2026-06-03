import argparse
import logging
import random
import numpy
import torch
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from sentence_transformers import (
    models,
    losses,
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator, SequentialEvaluator
from sentence_transformers.training_args import BatchSamplers
import os
from time import sleep
import torch.nn as nn
from glob import glob
import pickle
# Set environment variables for better network handling
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["STREAMING_READ_MAX_RETRIES"] = "20"
os.environ["STREAMING_READ_RETRY_INTERVAL"] = "5"

# Disable torch.compile to avoid Triton compilation issues with Python 3.13
os.environ["TORCH_COMPILE_DISABLE"] = "1"

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a sentence transformer model")
    parser.add_argument("--model", type=str, default="jhu-clsp/mmBERT-base",
                        help="Model name or path (default: jhu-clsp/mmBERT-base)")
    parser.add_argument("--output", type=str, default="mmbert-base-pretraining",
                        help="Output directory name (default: mmbert-base-pretraining)")
    parser.add_argument("--lr_type", type=str, default="LinearLR",
                        help="Learning rate scheduler type (default: LinearLR)")
    parser.add_argument("--data", type=str, default="datasets/step1",
                        help="Datapath directory (default: datasets/step1)")
    parser.add_argument("--loss_function", type=str, default="CachedMultipleNegativesRankingLoss",
                        help="Loss function to use (default: CachedMultipleNegativesRankingLoss)")
    parser.add_argument("--max_seq_length", type=int, default=1028,
                        help="Maximum sequence length (default: 1028)")
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        help="Optimizer to use (default: adamw_torch)")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Batch size per device (default: 48)")
    parser.add_argument("--mini_batch_size", type=int, default=32,
                        help="Mini batch size per device (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Learning rate (default: 4e-5)")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature (default: 0.05)")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Warmup proportion (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--cls", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Use CLS pooling (default: True)")
    parser.add_argument("--gather_across_devices", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True,
                        help="Multiple GPU pooling (default: True)")
    parser.add_argument("--dataset_cache_dir", type=str, default="./cached_datasets",
                        help="Directory to cache processed datasets (default: ./cached_datasets)")
    parser.add_argument("--force_reload", action="store_true",
                        help="Force reload dataset even if cache exists")
    parser.add_argument("--dataset", type=str, default="pre",
                        help="Use 'pre' for pre-training dataset, 'align' for alignment, or 'pre_align' for pre+alignment dataset (default: pre)")
    args = parser.parse_args()

    logging.info(f"Model: {args.model}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Max sequence length: {args.max_seq_length}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Mini batch size: {args.mini_batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"Warmup proportion: {args.warmup_proportion}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Gather across devices: {args.gather_across_devices}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Loss function: {args.loss_function}")
    logging.info(f"Optimizer: {args.optimizer}")
    logging.info(f"Learning rate scheduler: {args.lr_type}")

    logging.info("Pooling mode: CLS" if args.cls else "Pooling mode: mean")
    if args.cls:
        transformer = models.Transformer(args.model, max_seq_length=args.max_seq_length)
        dimension = transformer.get_word_embedding_dimension()
        pooling = models.Pooling(dimension, pooling_mode="cls")
        normalize = models.Normalize()
        model = SentenceTransformer(modules=[transformer, pooling, normalize])
        
    else:
        model = SentenceTransformer(
            args.model,
        ) 
        model.max_seq_length = args.max_seq_length
    
    # Check for existing checkpoint to resume training
    run_name = args.output
    checkpoint_path = None
    output_dir = f"models/{run_name}"
    if os.path.exists(output_dir):
        # Look for checkpoint directories (e.g., checkpoint-1000, checkpoint-2000)
        checkpoints = glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoints:
            # Sort by step number and get the latest
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint_path = checkpoints[-1]
            logging.info(f"Found checkpoint: {checkpoint_path}")
            logging.info("Resuming training from checkpoint...")
            logging.info("Note: ignore_data_skip is enabled to avoid long hang times with large datasets")
        else:
            logging.info("No checkpoint found. Starting training from scratch...")
    else:
        logging.info("Output directory does not exist. Starting training from scratch...")

    logging.info(f"Datasets mode: {args.dataset}...")

    # Before the loop, to load if exists
    cache_file = os.path.join("datasets/step1/", 'concate_dataset_all.pkl')
    if os.path.exists(cache_file) and not args.force_reload:
        concate_dataset = pickle.load(open(cache_file, 'rb'))
    else:
        concate_dataset = {}
        all_file = ["datasets/step1/c4.jsonl", "datasets/step1/parallel-sentences-ccmatrix.jsonl","datasets/step1/s2orc.jsonl", "datasets/step1/wikipedia.jsonl","datasets/step1/multilingual_cc_news.jsonl","datasets/step1/parallel-sentences-wikimatrix.jsonl","datasets/step1/parallel-sentences-tatoeba.jsonl","datasets/step1/parallel-sentences-jw300.jsonl","datasets/step1/parallel-sentences-opensubtitles.jsonl","datasets/step1/parallel-sentences-opus-100.jsonl","datasets/step1/parallel-sentences-talks.jsonl","datasets/step1/finetranslation.jsonl","datasets/step1/kalm_pretraining.jsonl"]
        for file in all_file:
            try:
                logging.info(f"Loading dataset from {file}...")
                file_name = file.split("/")[-1].replace(".jsonl", "")
                train_dataset = load_dataset('json', data_files=file, split='train')
                concate_dataset.update({file_name: train_dataset})
            except Exception as e:
                logging.warning(f"Failed to load dataset from {file}: {e}")
            # Save concate_dataset using pickle
        pickle.dump(concate_dataset, open(os.path.join("datasets/step1/", 'concate_dataset_all.pkl'), 'wb'))

    train_loss = losses.MultipleNegativesSymmetricRankingLossReweighting(
        model=model,
        scale=(1/args.temperature),
        gather_across_devices=args.gather_across_devices,
    )


    # 5. (Optional) Specify training arguments
    

    # Calculate total training steps for warmup_stable_decay scheduler
    total_dataset_size = sum(len(ds) for ds in concate_dataset.values())
    num_gpus = torch.cuda.device_count()
    gradient_accumulation_steps = 1  # Change if you use gradient accumulation
    steps_per_epoch = total_dataset_size // (args.batch_size * num_gpus * gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    
    # Calculate scheduler steps
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    num_decay_steps = int(total_steps * 0.1)  # Last 10% for decay
    num_stable_steps = total_steps - num_warmup_steps - num_decay_steps
    
    logging.info(f"Total dataset size: {total_dataset_size}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total training steps: {total_steps}")
    logging.info(f"Warmup steps: {num_warmup_steps} ({args.warmup_proportion*100}%)")
    # logging.info(f"Stable steps: {num_stable_steps} ({num_stable_steps/total_steps*100:.1f}%)")
    # logging.info(f"Decay steps: {num_decay_steps} (10%)")

    # Determine learning rate scheduler configuration
    if args.lr_type == "LinearLR":
        lr_scheduler_type = "linear"
        lr_scheduler_kwargs = None
    elif args.lr_type == "DecayLR":
        lr_scheduler_type = "warmup_stable_decay"
        lr_scheduler_kwargs = {
            "num_stable_steps": num_stable_steps,
            "num_decay_steps": num_decay_steps,
            "min_lr_ratio": 0.0,
        }
    elif args.lr_type == "CosineLR":
        lr_scheduler_type = "cosine"
        lr_scheduler_kwargs = None
    elif args.lr_type == "FlatternLR":
        lr_scheduler_type = "constant"
        lr_scheduler_kwargs = None
    else:
        raise ValueError(f"Unsupported lr_type: {args.lr_type}")

    # Build training arguments dictionary
    training_args_dict = {
        # Required parameter:
        "output_dir": f"models/{run_name}",
        # Optional training parameters:
        # "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        # gradient_accumulation_steps=16,   # Increased to maintain effective batch size
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_proportion,
        "lr_scheduler_type": lr_scheduler_type,
        "optim": args.optimizer,  # Use adamw_torch or muon
        # "optim": args.optimizer,  # Use adamw_torch or muon
        # optim="muon",  # Use adamw_torch or muon
        "adam_epsilon": 1e-12,  # Set epsilon to 1e-12
        "fp16": False,  # Set to False if you get an error that your GPU can't run on FP16
        "bf16": True,  # Set to True if you have a GPU that supports BF16
        "gradient_checkpointing": False,  # Disabled due to conflicts with LoRA + ZeRO Stage 3
        "dataloader_pin_memory": True,  # Reduce memory usage
        "remove_unused_columns": False,  # Keep all columns to avoid extra processing
        "dataloader_num_workers": 4,  # Increase to 16 workers per GPU for better throughput
        "dataloader_prefetch_factor": 2,  # Prefetch 16 batches per worker (deeper pipeline)
        "dataloader_persistent_workers": True,  # Keep workers alive between epochs
        "ignore_data_skip": True,  # Skip data iteration on checkpoint resume to avoid long hang times
        "torch_compile": False,  # Disable torch.compile due to Python 3.13 compatibility
        "torch_empty_cache_steps": 100,  # Clear CUDA cache every 100 steps to prevent fragmentation
        # deepspeed="deepspeed_zero3_config.json",  # Using ZeRO Stage 3 for better LoRA + multi-node stability
        # Optional tracking/debugging parameters:
        # eval_strategy="steps",
        # eval_steps=1000,  # Increased to reduce eval overhead
        "save_strategy": "steps",
        "save_steps": 1000,  # Increased to reduce save overhead
        "save_total_limit": 1,  # Reduce number of checkpoints to save memory
        "logging_steps": 10,   # Log metrics every 10 steps for better visibility in wandb
        "logging_first_step": True,  # Log the first step
        "report_to": "wandb",  # Enable Weights & Biases logging
        "run_name": run_name,  # Will be used in W&B
    }
    
    # Add lr_scheduler_kwargs if needed
    if lr_scheduler_kwargs is not None:
        training_args_dict["lr_scheduler_kwargs"] = lr_scheduler_kwargs
    
    training_args = SentenceTransformerTrainingArguments(**training_args_dict)
    # 6. (Optional) Create an evaluator & evaluate the base model
    # queries = eval_dataset["query"][:1000]
    # eval_queries = {query_id: query for query_id, query in enumerate(queries)}
    # corpus = eval_dataset["positive"]
    # eval_corpus = {doc_id: doc for doc_id, doc in enumerate(corpus)}
    # eval_relevant_docs = {index: [index] for index in range(len(queries))}
    # dev_evaluator = InformationRetrievalEvaluator(
    #     queries=eval_queries,
    #     corpus=eval_corpus,
    #     relevant_docs=eval_relevant_docs,
    #     batch_size=16,
    #     name="msmarco-eval-1kq-1kd",
    # )
    # nano_beir_evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)
    # dev_evaluator = SequentialEvaluator([dev_evaluator, nano_beir_evaluator])

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=concate_dataset,
        # eval_dataset=eval_dataset,
        loss=train_loss,
        # evaluator=dev_evaluator,
    )
    
    
    
    # Train with checkpoint resumption if available
    if checkpoint_path:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # 8. Evaluate the model performance again after training
    # dev_evaluator(model)

    # 9. Save the trained model
    model.save_pretrained(f"models/{run_name}/final")
    print(f"Model saved to models/{run_name}/final")

    # 10. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub(run_name)


if __name__ == "__main__":
    main()