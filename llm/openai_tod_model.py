import time
from typing import List, Tuple, Union, Optional
from copy import deepcopy
import time
import pickle
import json
from timeout_decorator import timeout, TimeoutError

import openai

from tod_model_base import TODModelBase
from llm.prompts import PromptFormater
from data_utils import (
    default_belief_state,
    default_book_state,
    context_list2str,
    domain_state_str2dict,
)

def call_openai_api(model_name: str, prompt: str, max_tokens: int) -> str:
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return completion.choices[0].message["content"].strip()
        
        except openai.error.RateLimitError as e:
            print(e)
            print("Rate limit exceeded. Waiting for 10 seconds...")
            time.sleep(10)
            continue

        except Exception as e:
            print(e)
            return ""


def call_openai_api_with_timeout(model_name: str, prompt: str, max_tokens: int) -> str:
    @timeout(60)
    def _call_openai_api(model_name: str, prompt: str, max_tokens: int) -> str:
        return call_openai_api(model_name=model_name,
                               prompt=prompt,
                               max_tokens=max_tokens)
    
    try:
        return _call_openai_api(model_name=model_name,
                                prompt=prompt,
                                max_tokens=max_tokens)
    except TimeoutError:
        print(f"TimeoutError: {prompt}")
        return "応答に時間がかかりすぎています。お手数ですが、もう一度お願いします。"

class OpenAITODModel(TODModelBase):
    def __init__(self, openai_model_name: str, max_context_turns: int, max_output_length: int, user_utterance_prefix: str, system_utterance_prefix: str,
                 state_prefix: str, db_result_prefix: str, max_candidate_entities: int, book_result_prefix: str):
        self.model_name = openai_model_name
        self.max_tokens = max_output_length
        self.prompt_formater = PromptFormater(
            max_context_turns=max_context_turns,
            user_utterance_prefix=user_utterance_prefix,
            system_utterance_prefix=system_utterance_prefix,
            state_prefix=state_prefix,
            db_result_prefix=db_result_prefix,
            max_candidate_entities=max_candidate_entities,
            book_result_prefix=book_result_prefix,
        )
        self.max_context_turns = max_context_turns
        
    def set_memory(self, memory: Optional[dict]) -> None:
        if memory is None:
            self.previous_belief_state = deepcopy(default_belief_state)
            self.previous_book_state = deepcopy(default_book_state)
        else:
            self.previous_belief_state = deepcopy(memory["previous_belief_state"])
            self.previous_book_state = deepcopy(memory["previous_book_state"])

    def get_memory(self) -> dict:
        return {
            "previous_belief_state": deepcopy(self.previous_belief_state),
            "previous_book_state": deepcopy(self.previous_book_state),
        }

    def _prepare_fewshot_examples(self, context: List[Tuple[str, str]]) -> List[dict]:
        raise NotImplementedError
    
    def predict_state(self, context: List[Tuple[str, str]], return_prompt: bool = False) -> dict:
        # 0. Retrieve few-shot examples
        fewshot_examples = self._prepare_fewshot_examples(context=context)

        # 1. Track state for 'general' domain first to get active domain
        general_domain_prompt = self.prompt_formater.make_state_prompt(domain="general",
                                                                       context=context,
                                                                       fewshot_examples=fewshot_examples)
        general_domain_state_str = call_openai_api_with_timeout(model_name=self.model_name,
                                                                prompt=general_domain_prompt,
                                                                max_tokens=self.max_tokens)
        general_damain_state, _ = domain_state_str2dict(domain="general", domain_state_str=general_domain_state_str)

        active_domain = general_damain_state["active_domain"]
        if not active_domain:
            active_domain = "general"

        # 2. Track state for active domain
        active_domain_prompt = self.prompt_formater.make_state_prompt(domain=active_domain,
                                                                      context=context,
                                                                      fewshot_examples=fewshot_examples)
        active_domain_state_str = call_openai_api_with_timeout(model_name=self.model_name,
                                                               prompt=active_domain_prompt,
                                                               max_tokens=self.max_tokens)
        active_domain_state, active_book_state = domain_state_str2dict(domain=active_domain,
                                                                       domain_state_str=active_domain_state_str)

        # 3. Merge state
        belief_state = deepcopy(self.previous_belief_state)
        belief_state["general"].update(
            {slot: value for slot, value in general_damain_state.items() if value}
        )
        belief_state[active_domain].update(
            {slot: value for slot, value in active_domain_state.items() if value}
        )
        self.previous_belief_state = deepcopy(belief_state)
        
        book_state = deepcopy(self.previous_book_state)
        if active_domain in book_state:
            book_state[active_domain].update(
                {slot: value for slot, value in active_book_state.items() if value}
            )
        self.previous_book_state = deepcopy(book_state)

        if return_prompt:
            return belief_state, book_state, active_domain_prompt
        else:
            return belief_state, book_state
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: dict, book_state: dict,
                          db_result: dict, book_result: dict, return_prompt: bool = False) -> str:
        # 0. Retrieve few-shot examples
        fewshot_examples = self._prepare_fewshot_examples(context=context)
        
        # 1. Generate response for active domain
        active_domain = belief_state["general"]["active_domain"]
        if not active_domain:
            active_domain = "general"

        active_domain_prompt = self.prompt_formater.make_response_prompt(
            domain=active_domain,
            context=context,
            belief_state=belief_state,
            book_state=book_state,
            db_result=db_result,
            book_result=book_result,
            fewshot_examples=fewshot_examples,
        )
        response = call_openai_api_with_timeout(model_name=self.model_name,
                                                prompt=active_domain_prompt,
                                                max_tokens=self.max_tokens)
        
        if return_prompt:
            return response, active_domain_prompt
        else:
            return response

class OpenAIZeroShotTODModel(OpenAITODModel):
    def _prepare_fewshot_examples(self, context: List[Tuple[str, str]]) -> None:
        return None

class OpenAIFewShotTODModel(OpenAITODModel):
    def __init__(self, faiss_db_fprefix: str, num_fewshot_examples: int, **kwargs):
        super().__init__(**kwargs)
        self.faiss_vs = pickle.load(open(f"{faiss_db_fprefix}.pkl", 'rb'))
        self.faiss_db_args = json.load(open(f"{faiss_db_fprefix}.args.json"))
        self.num_fewshot_examples = num_fewshot_examples
        
    def _prepare_fewshot_examples(self, context: List[Tuple[str, str]]) -> List[dict]:
        # Retrieve few-shot examples from faiss db
        context_str = context_list2str(
            context=context,
            max_context_turns=self.faiss_db_args["context_turns"],
            user_utterance_prefix=self.faiss_db_args["user_utterance_prefix"],
            system_utterance_prefix=self.faiss_db_args["system_utterance_prefix"],
        )
        
        retrieved_docs = self.faiss_vs.similarity_search(
            query=context_str,
            k=self.num_fewshot_examples,
        )
        retrieved_examples = []
        for doc in retrieved_docs:
            retrieved_examples.append({
                "context": doc.metadata["context"],
                "belief_state": doc.metadata["turn"]["dialogue_state"]["belief_state"],
                "book_state": doc.metadata["turn"]["dialogue_state"]["book_state"],
                "db_result": doc.metadata["turn"]["dialogue_state"]["db_result"],
                "book_result": doc.metadata["turn"]["dialogue_state"]["book_result"],
                "response": doc.metadata["turn"]["utterance"],
            })
        return retrieved_examples
