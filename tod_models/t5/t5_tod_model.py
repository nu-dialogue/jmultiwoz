from typing import List, Tuple, Union, Optional

from tod_models.tod_model_base import TODModelBase
from tod_models.t5.t5_generation_utils import create_t5_generation_function

from utils.data_utils import (
    context_list2str,
    state_dict2str,
    state_str2dict,
    db_result_dict2str,
    book_result_dict2str,
)

class T5TODModel(TODModelBase):
    def __init__(self, model_name_or_path: str, device: Union[str, int], use_background_generation_server: bool,
                 max_context_turns: int, max_input_length: str, max_output_length: str,
                 dst_task_prefix: str, rg_task_prefix: str, user_utterance_prefix: str, system_utterance_prefix: str,
                 state_prefix: str, db_result_prefix: str, max_candidate_entities: int, book_result_prefix: str):
        
        self.generation_fn = create_t5_generation_function(
            model_name_or_path=model_name_or_path,
            device=device,
            use_background_generation_server=use_background_generation_server,
        )

        self.max_context_turns = max_context_turns
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.dst_task_prefix = dst_task_prefix
        self.rg_task_prefix = rg_task_prefix
        self.user_utterance_prefix = user_utterance_prefix
        self.system_utterance_prefix = system_utterance_prefix
        self.state_prefix = state_prefix
        self.db_result_prefix = db_result_prefix
        self.max_candidate_entities = max_candidate_entities
        self.book_result_prefix = book_result_prefix

    def set_memory(self, memory: Optional[dict]) -> None:
        pass

    def get_memory(self) -> None:
        return None

    def predict_state(self, context: List[Tuple[str, str]]) -> Tuple[dict, dict]:
        context_str = context_list2str(
            context=context,
            max_context_turns=self.max_context_turns,
            user_utterance_prefix=self.user_utterance_prefix,
            system_utterance_prefix=self.system_utterance_prefix
        )

        input_text = f"{self.dst_task_prefix} {context_str}"
        state_str = self.generation_fn(
            input_text=input_text,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            do_sample=False, top_p=1.0, num_beams=1
        )
        belief_state, book_state = state_str2dict(state_str)
        return belief_state, book_state
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: dict, book_state: dict,
                          db_result: str, book_result: dict) -> str:
        context_str = context_list2str(
            context=context,
            max_context_turns=self.max_context_turns,
            user_utterance_prefix=self.user_utterance_prefix,
            system_utterance_prefix=self.system_utterance_prefix
        )
        state_str = state_dict2str(
            belief_state=belief_state,
            book_state=book_state
        )
        db_result_str = db_result_dict2str(
            db_result=db_result,
            max_candidate_entities=self.max_candidate_entities
        )
        book_result_str = book_result_dict2str(
            book_result=book_result
        )
        input_text = (f"{self.rg_task_prefix} {context_str} "
                      f"{self.state_prefix} {state_str} "
                      f"{self.db_result_prefix} {db_result_str} "
                      f"{self.book_result_prefix} {book_result_str}")
        
        output_text = self.generation_fn(
            input_text=input_text,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            do_sample=False, top_p=1.0, num_beams=1
        )
        return output_text
