from typing import List, Tuple, Union
from copy import deepcopy
import openai

from tod_model_base import TODModelBase
from llm.prompts import PromptFormater
from data_utils import (
    default_belief_state,
    default_book_state,
    domain_state_str2dict,
)

def call_openai_api(model_name: str, prompt: str, max_tokens: int) -> str:
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
    except openai.error.OpenAIError as e:
        print(e)
        return ""

class OpenAIZeroShotTODModel(TODModelBase):
    def __init__(self, openai_model_name: str, max_output_length: int, user_utterance_prefix: str, system_utterance_prefix: str,
                 state_prefix: str, db_result_prefix: str, max_candidate_entities: int, book_result_prefix: str, response_prefix: str):
        self.model_name = openai_model_name
        self.max_tokens = max_output_length
        self.prompt_formater = PromptFormater(
            user_utterance_prefix=user_utterance_prefix,
            system_utterance_prefix=system_utterance_prefix,
            state_prefix=state_prefix,
            db_result_prefix=db_result_prefix,
            max_candidate_entities=max_candidate_entities,
            book_result_prefix=book_result_prefix,
            response_prefix=response_prefix,
        )
        
    def init_session(self):
        self.previous_belief_state = deepcopy(default_belief_state)
        self.previous_book_state = deepcopy(default_book_state)
    
    def predict_state(self, context: List[Tuple[str, str]], return_prompt: bool = False) -> dict:
        # 1. Track state for 'general' domain first to get active domain
        general_domain_prompt = self.prompt_formater.make_state_prompt(domain="general", context=context)
        general_domain_state_str = call_openai_api(model_name=self.model_name,
                                                   prompt=general_domain_prompt,
                                                   max_tokens=self.max_tokens)
        general_damain_state, _ = domain_state_str2dict(domain="general", domain_state_str=general_domain_state_str)

        active_domain = general_damain_state["active_domain"]
        if not active_domain:
            active_domain = "general"

        # 2. Track state for active domain
        active_domain_prompt = self.prompt_formater.make_state_prompt(domain=active_domain, context=context)
        active_domain_state_str = call_openai_api(model_name=self.model_name,
                                                  prompt=active_domain_prompt,
                                                  max_tokens=self.max_tokens)
        active_domain_state, active_book_state = domain_state_str2dict(domain=active_domain, domain_state_str=active_domain_state_str)

        # 3. Merge state
        belief_state = deepcopy(self.previous_belief_state)
        belief_state.update({
            "general": general_damain_state,
            active_domain: active_domain_state,
        })
        self.previous_belief_state = deepcopy(belief_state)
        
        book_state = deepcopy(self.previous_book_state)
        if active_domain in book_state:
            book_state.update({
                active_domain: active_book_state,
            })
        self.previous_book_state = deepcopy(book_state)

        if return_prompt:
            return belief_state, book_state, active_domain_prompt
        else:
            return belief_state, book_state
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: dict, book_state: dict,
                          db_result: dict, book_result: dict, return_prompt: bool = False) -> str:
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
        )
        response = call_openai_api(model_name=self.model_name,
                                   prompt=active_domain_prompt,
                                   max_tokens=self.max_tokens)
        
        if return_prompt:
            return response, active_domain_prompt
        else:
            return response
