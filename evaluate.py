import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from sacrebleu import sentence_bleu

from transformers import (
    HfArgumentParser
)
from data_utils import (
    state_dict2str
)
from jmultiwoz import JMultiWOZDataset

@dataclass
class EvaluationArguments:
    dataset_dpath: str = field(
        metadata={"help": "Path to dataset directory."}
    )
    inference_output_dpath: str = field(
        metadata={"help": "Path to inference output directory."}
    )
    task_name: str = field(
        metadata={"help": ("The name of the task to evaluate. Select from: 'e2e' (End-to-End Generation) and "
                           "'rg' (Response Generation from Dialogue State).")}
    )

def dialogues_to_df(dialogues: dict) -> pd.DataFrame:
    turn_dicts = []
    for dialogue_name, dialogue in dialogues.items():
        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER":
                continue
            turn_dicts.append({
                "dialogue_name": dialogue_name,
                "turn_id": turn["turn_id"],
                "belief_state": turn.get("dialogue_state", {}).get("belief_state"),
                "book_state": turn.get("dialogue_state", {}).get("book_state"),
                "response": turn["utterance"],
            })
    return pd.DataFrame(turn_dicts).set_index(["dialogue_name", "turn_id"])

def compute_joint_goal_accuracy(row: pd.Series):
    state_str_ref = state_dict2str(belief_state=row["belief_state"], book_state=row["book_state"])
    state_str_hyp = state_dict2str(belief_state=row["hyp/belief_state"], book_state=row["hyp/book_state"])

    state_set_ref = set(state_str_ref.split(", "))
    state_set_hyp = set(state_str_hyp.split(", "))
    return int(state_set_ref == state_set_hyp)

def compute_slot_f1(row: pd.Series):
    state_str_ref = state_dict2str(belief_state=row["belief_state"], book_state=row["book_state"])
    state_str_hyp = state_dict2str(belief_state=row["hyp/belief_state"], book_state=row["hyp/book_state"])

    state_set_ref = set(state_str_ref.split(", "))
    state_set_hyp = set(state_str_hyp.split(", "))
    
    tp = len(state_set_ref & state_set_hyp)
    fp = len(state_set_hyp - state_set_ref)
    fn = len(state_set_ref - state_set_hyp)
    
    if tp == 0:
        return 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

def compute_response_bleu(row: pd.Series):
    response_ref = row["response"]
    response_hyp = row["hyp/response"]
    return sentence_bleu(hypothesis=response_hyp, references=[response_ref], tokenize="ja-mecab").score

def main():
    parser = HfArgumentParser((EvaluationArguments))
    eval_args = parser.parse_args_into_dataclasses()[0]

    dialogues_ref = JMultiWOZDataset(dataset_dpath=eval_args.dataset_dpath).get_dialogues(split="test")
    df_ref = dialogues_to_df(dialogues_ref)

    dialogues_hyp = json.load(
        open(os.path.join(eval_args.inference_output_dpath, f"{eval_args.task_name}.inference.json"))
    )
    df_hyp = dialogues_to_df(dialogues_hyp)

    assert df_ref.index.equals(df_hyp.index), "Reference and hypothesis dataframes have different indices."

    df = df_ref.merge(df_hyp.add_prefix("hyp/"), left_index=True, right_index=True)

    results = pd.DataFrame(index=df.index)
    if eval_args.task_name == "e2e":
        print("Computing joint goal accuracy...")
        results["joint_goal_accuracy"] = df.apply(compute_joint_goal_accuracy, axis=1)

        print("Computing slot F1...")
        results["slot_f1"] = df.apply(compute_slot_f1, axis=1)

        print("Computing response BLEU...")
        results["response_bleu"] = df.apply(compute_response_bleu, axis=1)
    
    elif eval_args.task_name == "rg":
        print("Computing response BLEU...")
        results["response_bleu"] = df.apply(compute_response_bleu, axis=1)

    else:
        raise ValueError(f"Invalid task name: {eval_args.task_name}")
    
    results.to_csv(
        os.path.join(eval_args.inference_output_dpath, f"{eval_args.task_name}.scores.csv")
    )
    results.describe().to_csv(
        os.path.join(eval_args.inference_output_dpath, f"{eval_args.task_name}.scores_summary.csv")
    )

if __name__ == "__main__":
    main()
