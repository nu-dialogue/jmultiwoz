from typing import List, Tuple, Union

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)

from tod_model_base import TODModelBase

class T5TODModel(TODModelBase):
    def __init__(self, model_name_or_path: str, device: Union[str, int], max_input_length: str, max_output_length: str,
                 dst_task_prefix: str, rg_task_prefix: str, user_utterance_prefix: str, system_utterance_prefix: str,
                 belief_state_prefix: str, db_result_prefix: str, book_result_prefix: str):
        
        self.device = device
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config).to(self.device)
        
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.dst_task_prefix = dst_task_prefix
        self.rg_task_prefix = rg_task_prefix
        self.user_utterance_prefix = user_utterance_prefix
        self.system_utterance_prefix = system_utterance_prefix
        self.belief_state_prefix = belief_state_prefix
        self.db_result_prefix = db_result_prefix
        self.book_result_prefix = book_result_prefix

    def generate(self, input_text: str, do_sample: bool = True, top_p: float = 0.95, num_beams: float = 5) -> str:
        default_trunction_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        model_inputs = self.tokenizer([input_text], max_length=self.max_input_length, truncation=True, return_tensors="pt")
        self.tokenizer.truncation_side = default_trunction_side

        outputs = self.t5_model.generate(
            input_ids=model_inputs.input_ids.to(self.device),
            attention_mask=model_inputs.attention_mask.to(self.device),
            max_length=self.max_output_length,
            do_sample=do_sample,
            top_p=top_p,
            num_beams=num_beams,
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
    
    def predict_state(self, context: List[Tuple[str, str]]) -> str:
        speaker2prefix = {"USER": self.user_utterance_prefix,
                          "SYSTEM": self.system_utterance_prefix}
        context = " ".join([f"{speaker2prefix[speaker]} {utterance}" for speaker, utterance in context])

        input_text = f"{self.dst_task_prefix} {context}"
        output_text = self.generate(input_text, do_sample=False, top_p=1.0, num_beams=5)
        return output_text
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: str, db_result: str, book_result: str) -> str:
        speaker2prefix = {"USER": self.user_utterance_prefix,
                          "SYSTEM": self.system_utterance_prefix}
        context = " ".join([f"{speaker2prefix[speaker]} {utterance}" for speaker, utterance in context])

        input_text = (f"{self.rg_task_prefix} {context} "
                      f"{self.belief_state_prefix} {belief_state} "
                      f"{self.db_result_prefix} {db_result} "
                      f"{self.book_result_prefix} {book_result}")
        output_text = self.generate(input_text, do_sample=True, top_p=0.95, num_beams=1)
        return output_text