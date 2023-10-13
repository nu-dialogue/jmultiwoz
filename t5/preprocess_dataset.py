import os
import sys
import json
import argparse
from tqdm import tqdm

from jmultiwoz import JMultiWOZDataset

from data_utils import (
    context_list2str,
    state_dict2str,
    db_result_dict2str,
    book_result_dict2str
)

def main(args):
    dataset = JMultiWOZDataset(dataset_dpath=args.dataset_dpath)

    os.makedirs(args.preprocessed_dpath, exist_ok=True)

    for split_name in ["train", "dev", "test"]:
        jsonlines = []
        dialogue_names = dataset.list_dialogues(split=split_name)
        for dialogue_name in tqdm(dialogue_names, desc=f"Processing {split_name}"):
            for context, turn in dataset.iter_dialogue_turns(split=split_name, dialogue_name=dialogue_name):
                context_str = context_list2str(
                    context=context,
                    max_context_turns=args.max_context_turns,
                    user_utterance_prefix=args.user_utterance_prefix,
                    system_utterance_prefix=args.system_utterance_prefix
                )
                state_str = state_dict2str(
                    belief_state=turn["dialogue_state"]["belief_state"],
                    book_state=turn["dialogue_state"]["book_state"]
                )
                db_result_str = db_result_dict2str(
                    db_result=turn["dialogue_state"]["db_result"],
                    max_candidate_entities=args.max_candidate_entities
                )
                book_result_str = book_result_dict2str(
                    book_result=turn["dialogue_state"]["book_result"]
                )
                response = turn["utterance"]

                # Dialogue State Tracking (DST) task
                jsonline_dst = json.dumps({
                    "input_text": f"{args.dst_task_prefix} {context_str}",
                    "output_text": state_str,
                }, ensure_ascii=False)

                # Response Generation (RG) task
                jsonline_rg = json.dumps({
                    "input_text": (
                        f"{args.rg_task_prefix} {context_str} "
                        f"{args.state_prefix} {state_str} "
                        f"{args.db_result_prefix} {db_result_str} "
                        f"{args.book_result_prefix} {book_result_str}"
                    ),
                    "output_text": response,
                }, ensure_ascii=False)

                jsonlines += [
                    jsonline_dst + "\n",
                    jsonline_rg + "\n"
                ]
        with open(os.path.join(args.preprocessed_dpath, f"{split_name}.json"), "w") as f:
            f.writelines(jsonlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dpath", type=str, required=True,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--preprocessed_dpath", type=str, required=True,
                        help="Path to the directory to save the preprocessed dataset.")

    parser.add_argument("--max_context_turns", type=int, default=0,
                        help="Number of context turns to use. Set to 0 to use all context turns.")
    parser.add_argument("--dst_task_prefix", type=str, default="対話から信念状態を推定:", 
                        help="A prefix to add before every input text of dialogue state tracking (DST) task.")
    parser.add_argument("--rg_task_prefix", type=str, default="対話から応答を生成:",
                        help="A prefix to add before every input text of response generation (RG) task.")
    parser.add_argument("--user_utterance_prefix", type=str, default="<顧客>",
                        help="A prefix to add before every user utterance text.")
    parser.add_argument("--system_utterance_prefix", type=str, default="<店員>",
                        help="A prefix to add before every system utterance text.")
    parser.add_argument("--state_prefix", type=str, default="<信念状態>",
                        help="A prefix to add before every belief state text.")
    parser.add_argument("--db_result_prefix", type=str, default="<検索結果>",
                        help="A prefix to add before every db result text.")
    parser.add_argument("--book_result_prefix", type=str, default="<予約結果>",
                        help="A prefix to add before every book result text.")
    parser.add_argument("--max_candidate_entities", type=int, default=3,
                        help="Number of candidate entities to show in db result.")
    
    args = parser.parse_args()
    main(args)
