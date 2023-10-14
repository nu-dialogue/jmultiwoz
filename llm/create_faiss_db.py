import argparse
import pickle
import sys
import os
import json
from copy import deepcopy

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.docstore.document import Document

from jmultiwoz import JMultiWOZDataset
from data_utils import (
    context_list2str,
)

def main(args):
    if args.embeddings == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=args.model)
    elif args.embeddings == 'openai':
        embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                      query_model_name=args.model,
                                      openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
    else:
        raise ValueError(f"Unknown embeddings: {args.embeddings}")
        
    dataset = JMultiWOZDataset(dataset_dpath=args.dataset_dpath)

    domain_counter = {domain: 0 for domain in dataset.available_domains}
    docs = []
    dialogue_names = dataset.list_dialogues(split=args.split)
    for dialogue_name in dialogue_names:
        print("Current data: ")
        print(", ".join([f"{domain}: {cnt:02d}" for domain, cnt in domain_counter.items()]))
        goal = dataset.get_dialogue(split=args.split, dialogue_name=dialogue_name)["goal"]

        if all([cnt >= args.dialogues_per_domain for cnt in domain_counter.values()]):
            # Break if all domains have enough dialogues
            break

        if all([domain_counter[domain] >= args.dialogues_per_domain for domain in goal]):
            # Skip if all domains in the goal have enough dialogues
            continue

        for domain in goal:
            domain_counter[domain] += 1

        for context, turn in dataset.iter_dialogue_turns(split=args.split, dialogue_name=dialogue_name):
            context_str = context_list2str(
                context=context,
                max_context_turns=args.context_turns,
                user_utterance_prefix=args.user_utterance_prefix,
                system_utterance_prefix=args.system_utterance_prefix,
            )
            docs.append(
                Document(
                    page_content=context_str,
                    metadata={
                        'dialogue_name': dialogue_name,
                        'context': context,
                        'turn': turn,
                    }
                )
            )
    
    faiss_vs = FAISS.from_documents(documents=docs, embedding=embeddings)

    print("Saving faiss db...")
    os.makedirs(os.path.dirname(args.output_faiss_db_fprefix), exist_ok=True)

    json.dump(args.__dict__, open(f"{args.output_faiss_db_fprefix}.args.json", 'w'), indent=4, ensure_ascii=False)
    json.dump(domain_counter, open(f"{args.output_faiss_db_fprefix}.domain_counter.json", 'w'), indent=4)
    with open(f"{args.output_faiss_db_fprefix}.pkl", 'wb') as f:
        pickle.dump(faiss_vs, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dpath', type=str, required=True,
                        help='Path to the directory of the dataset.')
    parser.add_argument('--output_faiss_db_fprefix', type=str, required=True,
                        help='Output path prefix for the faiss db.')
    parser.add_argument('--split', type=str, default="train",
                        help='Split name of the dataset (train/dev/test).')
    parser.add_argument('--context_turns', type=int, default=2,
                        help='Number of context turns to use.')
    parser.add_argument('--dialogues_per_domain', type=int, default=20,
                        help='Number of dialogues to use per domain.')
    parser.add_argument('--user_utterance_prefix', type=str, default='<顧客>',
                        help='Prefix for user utterance.')
    parser.add_argument('--system_utterance_prefix', type=str, default='<店員>',
                        help='Prefix for system utterance.')
    parser.add_argument('--embeddings', type=str, default='huggingface',
                        help='Embeddings to use (huggingface/openai).')
    parser.add_argument('--model', type=str, default='cl-nagoya/sup-simcse-ja-large',
                        help='Model name to use for embeddings.')
    
    args = parser.parse_args()
    main(args)
