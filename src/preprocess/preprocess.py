import argparse
import os
from datasets import load_dataset

def format_hotpot(sample):
    """
    Function that takes a single data sample derived from Huggingface datasets API: (https://huggingface.co/docs/datasets/index)
    and formats it into llama2 prompt format. For more information about llama2 prompt format, 
    please refer to https://huggingface.co/blog/llama2#how-to-prompt-llama-2 
    
    An example prompt is shown in the following:
    <s>
      [INST] <<SYS>>
        {{system}}
      <</SYS>>

      ### Question
      {{question}}

      ### Context
      {{context}}[/INST] {{answer}}</s>
    
    @type  sample: Dataset sample
    @param sample: dataset sample
    @rtype:   string
    @return:  llama2 prompt format
    """
    
    prefix = "<s>"
    postfix = "</s>"
    system_start_tag = "<<SYS>>"
    system_end_tag = "<</SYS>>"
    instruction_start_tag = "[INST]"
    instruction_end_tag = "[/INST]"
    context = "\n".join([ "".join(x) for x in sample['context']['sentences']])
    system = f"Given the following context, answer the question as accurately as possible:"
    question_prompt = f"### Question\n{sample['question']}"
    context_prompt = f"### Context\n{context}"
    prompt = f"{prefix}\n{instruction_start_tag} {system_start_tag}\n{system}\n{system_end_tag}\n\n{question_prompt}\n\n{context_prompt}{instruction_end_tag} {sample['answer']} {postfix}" 
    return prompt

# template dataset to add prompt to each sample
def template_dataset(sample):
    """
    Create a field for the given sample to store formatted llama2 prompt.
    
    @type  sample: Dataset sample
    @param sample: Dataset sample
    @rtype:   Dataset
    @return:  Dataset sample
    
    """
    sample["text"] = f"{format_hotpot(sample)}"
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset-name", type=str)
    parser.add_argument("--train-data-split", type=str, default=":10%")
    parser.add_argument("--eval-data-split", type=str, default=":10%")
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))    
    
    dataset = load_dataset(args.hf_dataset_name, "distractor", split=f"train[{args.train_data_split}]")
    new_dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
    training_input_path = "/opt/ml/processing/train"
    new_dataset.save_to_disk(training_input_path)
    print(f"training dataset uploaded to: {training_input_path}")

    eval_dataset = load_dataset(args.hf_dataset_name, "distractor", split=f"train[{args.eval_data_split}]")
    new_eval_dataset = dataset.map(template_dataset, remove_columns=list(eval_dataset.features))
    eval_input_path = "/opt/ml/processing/eval"
    new_eval_dataset.save_to_disk(eval_input_path)
    print(f"eval dataset uploaded to: {eval_input_path}")

    