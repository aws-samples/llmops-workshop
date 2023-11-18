import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_from_disk
import tarfile
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.collection import Collection
from tqdm import tqdm
from smexperiments_callback import SageMakerExperimentsCallback
import pathlib
from botocore.exceptions import ClientError
import logging
from urllib.parse import urlparse
import json
import shutil
import bitsandbytes as bnb


# Output directory where the model predictions and checkpoints will be stored
output_dir = "/opt/ml/model/"
base_model_path = "/tmp/basemodel"
model_eval_save_dir = "/tmp/eval"

# Load the entire model on the GPU 0
device_map = "auto"

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", 
                        help="Model id to use for training."),
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--fp16", type=bool, default=False,  help="Use fp16 for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="per device training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="per device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="number of batches to accumulate for gradients before optimization step")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, 
                        help="apply gradient checkpointing for moemory optimization at training time")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="gradient norm clipping value")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_32bit", help="optimizer to use")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="learning rate scheduler type")
    parser.add_argument("--max_steps", type=int, default=-1, help="maximum steps to train")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="learning rate warm up ratio")
    parser.add_argument("--group_by_length", type=bool, default=True, help="groupping datase by length")
    parser.add_argument("--save_steps", type=int, default=25, help="number of training steps before saving a checkpoint")
    parser.add_argument("--logging_steps", type=int, default=25, help="number of steps before logging the training metrics")
    parser.add_argument("--max_seq_length", type=int, default=None, help="maximum sequence length")
    parser.add_argument("--packing", type=bool, default=False, help="pack multiple examples into input sequence") 
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
    parser.add_argument("--use_4bit", type=bool, default=True, help="Whether to use 4bit quantization")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", help="compute dtype for bnb_4bit")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type")
    parser.add_argument("--use_nested_quant", type=bool, default=False, 
                        help="activate nested quantization for 4bit base models (double quantization")
    parser.add_argument("--lora_bias", type=str, default="none", 
                        help="whether to use bias in the lora adapter")
    parser.add_argument("--merge_weights", type=bool, default=True, 
                        help="Whether to merge LoRA weights with base model.")
    parser.add_argument("--base_model_group_name", type=str, default="None",
                        help="Optional base model group name.")
    parser.add_argument("--region", type=str, default="us-east-1",
                        help="the region where the training job is run.")
    parser.add_argument("--sm_train_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--sm_validation_dir", type=str, default="/opt/ml/input/data/validation")
    parser.add_argument("--model_eval_s3_loc", type=str, default="")
    parser.add_argument("--run_experiment", type=str, default="True")

    args = parser.parse_known_args()
    return args


def s3_download(s3_bucket, s3_object_key, local_file_name, s3_client=boto3.client('s3')):
    """
    Function that downloads an object from S3 into local filesystem using boto3 library.
    """
    meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    with tqdm(total=total_length,  
              desc=f'source: s3://{s3_bucket}/{s3_object_key}', 
              bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",  
              unit='B', 
              unit_scale=True, 
              unit_divisor=1024
             ) as pbar:
        with open(local_file_name, 'wb') as f:
            s3_client.download_fileobj(s3_bucket, s3_object_key, f, Callback=pbar.update)


def download_and_untar_s3_tar(destination_path, source_s3_path):
    """
    Function that downloads a file on S3, then untar the file in local filesystem.
    """
    src_s3_bucket = source_s3_path.split('/')[2]
    src_s3_prefix = "/".join(source_s3_path.split('/')[3:])
    destination_file_path = os.path.join(destination_path, os.path.basename(source_s3_path))
    print(f"Downloading file from {src_s3_bucket}/{src_s3_prefix} to {destination_file_path}")
    s3_download(
        s3_bucket=src_s3_bucket,
        s3_object_key=src_s3_prefix,
        local_file_name=destination_file_path
    )

    # Create a tarfile object and extract the contents to the local disk
    tar = tarfile.open(destination_file_path, "r")
    tar.extractall(path=destination_path)
    tar.close()

def model_data_uri_from_model_package(model_group_name, region="us-east-1"):
    """
    Function that retrieves the model artifact for the given model group name.
    """
    sagemaker_session = sagemaker.session.Session(boto3.session.Session(region_name=region))
    region = sagemaker_session.boto_region_name

    sm_client = boto3.client('sagemaker', region_name=region)

    model_packages = sm_client.list_model_packages(
        ModelPackageGroupName=model_group_name
    )['ModelPackageSummaryList']

    model_package_name = sorted(
        [
            (package['ModelPackageVersion'], package['ModelPackageArn']) 
            for package in model_packages if package['ModelApprovalStatus'] == 'Approved'],
        reverse=False
    )[-1][-1]
    print(f"found model package: {model_package_name}")

    return sm_client.describe_model_package(
        ModelPackageName=model_package_name
    )['InferenceSpecification']['Containers'][0]['ModelDataUrl']


def quantization_config(args, compute_dtype):
    """
    At a high level, QLoRA uses 4-bit quantization to compress a pretrained language model.
    This function sets up the quantization configuration for model training with QLoRA. 
    
    The LoRA layers are the only parameters being updated during training. 
    Read more about LoRA in the original LoRA paper (https://arxiv.org/abs/2106.09685). 
    
    """
    
    # 4 bit configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    return bnb_config


def load_pretrained_model(args, model_name, compute_dtype):
    """
    Loads the pretrained model into GPU memory for finetuning. 
    
    There are 2 modes supported: 
    1. Directly download a pretrained model weights
    from Huggingface Hub over the public internet.
    
    2. Download the pretrained model weight from a base model package
    registered in SageMaker Model Registry. Downloading weights from S3 
    could improve the download speed significantly. 
    
    """
    
    if args.base_model_group_name != "None":
        os.makedirs(base_model_path, exist_ok=True)
        model_data_uri = model_data_uri_from_model_package(
            model_group_name=args.base_model_group_name,
            region=args.region
        )
        download_and_untar_s3_tar(
        destination_path=base_model_path, 
        source_s3_path=model_data_uri
        )
    bnb_config = quantization_config(args, compute_dtype)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def load_tokenizer(args, model_name):
    """
    Loads the tokenizer for llama2 model. Tokenizer is used in the training process to 
    convert the input texts into tokens. 
    Please refer to: https://huggingface.co/docs/transformers/v4.31.0/model_doc/llama2#transformers.LlamaTokenizer 
    to learn more about this specific tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    return tokenizer

def load_lora_config(args, modules):
    """
    Loads QLoRA configuration for the training job. 
    
    """
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules = modules
    )
    return peft_config


def s3_upload(model_evaluation_s3_path, local_file_name, s3_client=boto3.client('s3')):
    """
    Uploads the given file in local file system to a specified S3 location.
    This function is used for uploading the model evaluation metrics to S3.
    The metrics will be used when registering the trained model with SageMaker Model Registry.
    """
    o = urlparse(model_evaluation_s3_path)
    s3_bucket = o.netloc
    s3_object_key = o.path
    local_base_file_name = os.path.basename(local_file_name)
    s3_object_key = os.path.join(s3_object_key, local_base_file_name)
    try:
        response = s3_client.upload_file(local_file_name, s3_bucket, s3_object_key.lstrip('/'))
    except ClientError as e:
        logging.error(e)


def model_evaluation(args, metrics):
    """
    Captures the training and evaluation metrics from the model training exercise. 
    The metrics will be written to file, and copied to the specifiued S3 locaiton.
    """
    train_loss = float('inf')
    eval_loss = float('inf')
    for metric in metrics:
        if 'train_loss' in metric:
            train_loss = metric['train_loss']
        elif 'eval_loss' in metric:
            eval_loss = metric['eval_loss']

    evaluation_metrics =  {
        "regression_metrics": {
            "train_loss" : {
                "value" : train_loss
            },
            "eval_loss" : {
                "value" : eval_loss
            }

        },
    }
    print(f"evaluation metrics: {evaluation_metrics}")
    #Save Evaluation Report
    pathlib.Path(model_eval_save_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{model_eval_save_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_metrics))
    s3_upload(args.model_eval_s3_loc, evaluation_path)

    
# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def training_function(args):
    print(f"merging weights: {args.merge_weights}")
    train_dataset = load_from_disk(args.sm_train_dir)
    eval_dataset = load_from_disk(args.sm_validation_dir)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    model_name = args.model_id
    if args.base_model_group_name != "None":
        model_name = base_model_path
    model = load_pretrained_model(args, model_name, compute_dtype)
    tokenizer = load_tokenizer(args, model_name)
    lora_modules = find_all_linear_names(model)
    lora_config = load_lora_config(args, lora_modules)
    packing = True if args.packing else False
    fp16 = True if args.fp16 else False
    bf16 = False
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            bf16=True

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optimizer,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps",
        report_to="tensorboard",
        evaluation_strategy="steps",
    )

    # Set supervised fine-tuning parameters
    if args.run_experiment == "True":
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
            callbacks=[SageMakerExperimentsCallback(region=args.region)]
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing
        )
    trainer.train()
    evaluation_metrics = trainer.evaluate()
    print(f"Training metrics: {trainer.state.log_history}")
    model_evaluation(args, trainer.state.log_history)

    if args.merge_weights:
        print(f"saving adapter weight combined with the base model weight")
        # merge adapter weights with base model and save
        # save int 4 model
        new_model = "/tmp/lora-adapter-weights"
        trainer.model.save_pretrained(new_model, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, new_model)
        model = model.merge_and_unload()
        
        model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.save_pretrained(output_dir)
        
        source_dir = './djl-inference/'

        # copy djl-inference files to model directory
        for f in os.listdir(source_dir):
            source_f = os.path.join(source_dir, f)
            
            # Copy the files to the destination folder
            shutil.copy(source_f, output_dir)

    else:
        print(f"saving adapter weights only")
        trainer.model.save_pretrained(output_dir, safe_serialization=True)

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
