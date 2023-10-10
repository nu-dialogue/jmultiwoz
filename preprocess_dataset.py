import os
import json
import argparse
from tqdm import tqdm

from data_utils import (
    state_dict2str,
    db_result_dict2str,
    book_result_dict2str
)

def make_dialogue_to_jsonlines(dialogue, args):
    dialogue_name = dialogue["dialogue_name"]
    goal = dialogue["goal"]

    jsonlines = []
    context = []
    for turn_id in range(0, len(dialogue["turns"]), 2):
        assert dialogue["turns"][turn_id]["speaker"] == "USER"

        if turn_id + 1 >= len(dialogue["turns"]):
            # End of dialogue at user turn
            break
        assert dialogue["turns"][turn_id + 1]["speaker"] == "SYSTEM"

        user_turn = dialogue["turns"][turn_id]
        system_turn = dialogue["turns"][turn_id + 1]

        context.append(("USER", user_turn["utterance"]))

        state_str = state_dict2str(
            belief_state=system_turn["dialogue_state"]["belief_state"], book_state=system_turn["dialogue_state"]["book_state"]
        )
        db_result_str = db_result_dict2str(
            db_result=system_turn["dialogue_state"]["db_result"], max_candidate_entities=args.max_candidate_entities
        )
        book_result_str = book_result_dict2str(
            book_result=system_turn["dialogue_state"]["book_result"]
        )
        response = system_turn["utterance"]

        jsonline = json.dumps({
            "dialogue_name": dialogue_name,
            "user_turn_id": turn_id,
            "system_turn_id": turn_id + 1,
            "context": context,
            "state_str": state_str,
            "db_result_str": db_result_str,
            "book_result_str": book_result_str,
            "response": response,
        }, ensure_ascii=False)
        jsonlines.append(jsonline + "\n")

        context.append(("SYSTEM", system_turn["utterance"]))
    
    return jsonlines


def main(args):
    split_list = json.load(open(os.path.join(args.dataset_dpath, "split_list.json")))
    dialogues = json.load(open(os.path.join(args.dataset_dpath, "dialogues.json")))

    os.makedirs(args.preprocessed_dpath, exist_ok=True)

    for split_name, dialogue_indices in split_list.items():
        jsonlines = []
        for dialogue_index in tqdm(dialogue_indices, desc=f"Preprocessing {split_name}"):
            dialogue = dialogues[dialogue_index]
            jsonlines += make_dialogue_to_jsonlines(dialogue, args)
        with open(os.path.join(args.preprocessed_dpath, f"{split_name}.json"), "w") as f:
            f.writelines(jsonlines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dpath", type=str, required=True)
    parser.add_argument("--preprocessed_dpath", type=str, required=True)
    parser.add_argument("--max_candidate_entities", type=int, default=3)
    
    args = parser.parse_args()

    main(args)