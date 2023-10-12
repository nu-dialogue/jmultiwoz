
import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

import multiprocessing as standard_mp
import torch.multiprocessing as torch_mp

from transformers import (
    HfArgumentParser
)
from jmultiwoz import JMultiWOZDataset, JMultiWOZDatabase

@dataclass
class TODModelArguments:

    tod_model_type: str = field(
        metadata={"help": "Type of TOD model to use. Select from: 't5', 'openai-zs', 'openai-fs'"}
    )
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    dst_task_prefix: Optional[str] = field(
        default="対話から信念状態を推定:",
        metadata={"help": "A prefix to add before every input text of dialogue state tracking (DST) task."},
    )
    rg_task_prefix: Optional[str] = field(
        default="対話から応答を生成:",
        metadata={"help": "A prefix to add before every input text of response generation (RG) task."},
    )

    user_utterance_prefix: Optional[str] = field(
        default="<顧客>",
        metadata={"help": "A prefix to add before every user utterance text."},
    )
    system_utterance_prefix: Optional[str] = field(
        default="<店員>",
        metadata={"help": "A prefix to add before every system utterance text."},
    )
    state_prefix: Optional[str] = field(
        default="<信念状態>",
        metadata={"help": "A prefix to add before every belief state text."},
    )
    db_result_prefix: Optional[str] = field(
        default="<検索結果>",
        metadata={"help": "A prefix to add before every db result text."},
    )
    book_result_prefix: Optional[str] = field(
        default="<予約結果>",
        metadata={"help": "A prefix to add before every book result text."},
    )
    max_candidate_entities: int = field(
        default=3,
        metadata={"help": "The maximum number of candidate entities to show in DB result."}
    )

    max_input_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_output_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

@dataclass
class InferenceArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    task_name: str = field(
        metadata={"help": ("The name of the task to evaluate. Select from: 'e2e' (End-to-End Generation) and "
                           "'rg' (Response Generation from Dialogue State).")}
    )
    dataset_dpath: str = field(
        default="dataset/JMultiWOZ_1.0",
        metadata={"help": "Path to the dataset directory."}
    )
    world_size: int = field(
        default=1,
        metadata={"help": "The number of processes to use for evaluation."}
    )

def load_tod_model(tod_model_args, device="cuda"):
    if tod_model_args.tod_model_type == "t5":
        from t5.t5_tod_model import T5TODModel
        tod_model = T5TODModel(
            model_name_or_path=tod_model_args.model_name_or_path,
            device=device,
            max_input_length=tod_model_args.max_input_length,
            max_output_length=tod_model_args.max_output_length,
            dst_task_prefix=tod_model_args.dst_task_prefix,
            rg_task_prefix=tod_model_args.rg_task_prefix,
            user_utterance_prefix=tod_model_args.user_utterance_prefix,
            system_utterance_prefix=tod_model_args.system_utterance_prefix,
            state_prefix=tod_model_args.state_prefix,
            db_result_prefix=tod_model_args.db_result_prefix,
            max_candidate_entities=tod_model_args.max_candidate_entities,
            book_result_prefix=tod_model_args.book_result_prefix,
        )
    elif tod_model_args.tod_model_type == "openai-zs":
        from llm.openai_tod_model import OpenAIZeroShotTODModel
        tod_model = OpenAIZeroShotTODModel(
            openai_model_name=tod_model_args.model_name_or_path,
            max_output_length=tod_model_args.max_output_length,
            user_utterance_prefix=tod_model_args.user_utterance_prefix,
            system_utterance_prefix=tod_model_args.system_utterance_prefix,
            state_prefix=tod_model_args.state_prefix,
            db_result_prefix=tod_model_args.db_result_prefix,
            max_candidate_entities=tod_model_args.max_candidate_entities,
            book_result_prefix=tod_model_args.book_result_prefix,
            response_prefix=tod_model_args.rg_task_prefix,
        )
    elif tod_model_args.tod_model_type == "openai-fs":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid tod_model_type: {tod_model_args.tod_model_type}")
    return tod_model

def e2e_inference(rank, tod_model_args, infer_args, dialogue_names_by_process, dataset, results_by_rank):
    database = JMultiWOZDatabase(db_dpath=os.path.join(infer_args.dataset_dpath, "database"))

    tod_model = load_tod_model(tod_model_args, device=rank)

    dialogue_names = dialogue_names_by_process[rank]
    results = {}
    print(f"Rank {rank} is processing {len(dialogue_names)} dialogues...")
    for dialogue_name in tqdm(dialogue_names):
        goal = dataset.get_dialogue_goal(split="test", dialogue_name=dialogue_name)
        results[dialogue_name] = {
            "dialogue_name": dialogue_name,
            "turns": [],
        }
        tod_model.init_session()
        for context, true_turn in dataset.iter_dialogue_turns(split="test", dialogue_name=dialogue_name):
            assert true_turn["speaker"] == "SYSTEM", "Must be system turn."

            # breakpoint()

            # 1. Dialogue State Tracking
            belief_state, book_state = tod_model.predict_state(
                context=context,
            )

            # 2. Get DB result
            db_result = database.get_db_result(
                belief_state=belief_state,
                goal=goal,
                oracle_db_result=true_turn["dialogue_state"]["db_result"],
            )

            # 3. Get Book result
            book_result = database.get_book_result(
                book_state=book_state,
                goal=goal,
                oracle_book_result=true_turn["dialogue_state"]["book_result"],
            )

            # 4. Generate response
            response = tod_model.generate_response(
                context=context,
                belief_state=belief_state,
                book_state=book_state,
                db_result=db_result,
                book_result=book_result,
            )

            results[dialogue_name]["turns"].append({
                "turn_id": true_turn["turn_id"],
                "speaker": "SYSTEM",
                "dialogue_state": {
                    "belief_state": belief_state,
                    "book_state": book_state,
                    "db_result": db_result,
                    "book_result": book_result,
                },
                "utterance": response,
            })
            # Save intermediate results temporarily
            jsonline = json.dumps({
                "dialogue_name": dialogue_name,
                **results[dialogue_name]["turns"][-1],
            }, ensure_ascii=False)
            with open(os.path.join(infer_args.output_dir, f"{infer_args.task_name}.inference.tmp.{rank}.jsonl"), "a") as f:
                f.write(jsonline + "\n")

    results_by_rank[rank] = results
    # return results

def rg_inference(rank, tod_model_args, infer_args, dialogue_names_by_process, dataset, results_by_rank):
    tod_model = load_tod_model(tod_model_args, device=rank)

    dialogue_names = dialogue_names_by_process[rank]
    results = {}
    print(f"Rank {rank} is processing {len(dialogue_names)} dialogues...")
    for dialogue_name in tqdm(dialogue_names):
        results[dialogue_name] = {
            "dialogue_name": dialogue_name,
            "turns": [],
        }
        tod_model.init_session()
        for context, true_turn in dataset.iter_dialogue_turns(split="test", dialogue_name=dialogue_name):
            assert true_turn["speaker"] == "SYSTEM", "Must be system turn."

            # Generate response from oracle state
            response = tod_model.generate_response(
                context=context,
                belief_state=true_turn["dialogue_state"]["belief_state"],
                book_state=true_turn["dialogue_state"]["book_state"],
                db_result=true_turn["dialogue_state"]["db_result"],
                book_result=true_turn["dialogue_state"]["book_result"],
            )

            results[dialogue_name]["turns"].append({
                "turn_id": true_turn["turn_id"],
                "speaker": "SYSTEM",
                "utterance": response,
            })
            # Save intermediate results temporarily
            jsonline = json.dumps({
                "dialogue_name": dialogue_name,
                **results[dialogue_name]["turns"][-1],
            }, ensure_ascii=False)
            with open(os.path.join(infer_args.output_dir, f"{infer_args.task_name}.inference.tmp.{rank}.jsonl"), "a") as f:
                f.write(jsonline + "\n")

    results_by_rank[rank] = results
    # return results

def main():
    parser = HfArgumentParser((TODModelArguments, InferenceArguments))
    tod_model_args, infer_args = parser.parse_args_into_dataclasses()

    os.makedirs(infer_args.output_dir, exist_ok=True)
    json.dump(
        {"tod_model_args": tod_model_args.__dict__, "infer_args": infer_args.__dict__},
        open(os.path.join(infer_args.output_dir, f"{infer_args.task_name}.args.json"), "w"),
        ensure_ascii=False,
        indent=4,
    )

    dataset = JMultiWOZDataset(
        dataset_dpath=infer_args.dataset_dpath
    )
    dialogue_names = dataset.list_dialogues(split="test")

    dialogue_names_by_process = {
        rank : names.tolist() for rank, names in enumerate(
            np.array_split(dialogue_names, infer_args.world_size)
        )
    }

    if infer_args.task_name == "e2e":
        infer_fn = e2e_inference
    elif infer_args.task_name == "rg":
        infer_fn = rg_inference
    else:
        raise ValueError(f"Invalid task_name: {infer_args.task_name}")

    if infer_args.world_size > 1:
        print(f"Spawning {infer_args.world_size} processes...")
        manager = standard_mp.Manager()
        results_by_rank = manager.dict()
        torch_mp.spawn(
            fn=infer_fn,
            args=(
                tod_model_args,
                infer_args,
                dialogue_names_by_process,
                dataset,
                results_by_rank
            ),
            nprocs=infer_args.world_size,
            join=True,
        )
        
    else:
        print("Computing on single process...")
        results_by_rank = {}
        infer_fn(
            rank=0,
            tod_model_args=tod_model_args,
            infer_args=infer_args,
            dialogue_names_by_process=dialogue_names_by_process,
            dataset=dataset,
            results_by_rank=results_by_rank,
        )
        
    results = {}
    for rank in range(infer_args.world_size):
        results.update(results_by_rank[rank])
    
    json.dump(
        results,
        open(os.path.join(infer_args.output_dir, f"{infer_args.task_name}.inference.json"), "w"),
        ensure_ascii=False,
        indent=4,
    )
    
if __name__ == "__main__":
    main()
