import os
import openai
import random, string
import json
from typing import List, Tuple
from copy import deepcopy

from jmultiwoz import JMultiWOZDataset, JMultiWOZDatabase

from tod_model_base import TODModelBase
from t5.t5_tod_model import T5TODModel
from llm.openai_tod_model import OpenAIFewShotTODModel

openai.api_key = os.environ.get('OPENAI_API_KEY')

TOD_MODEL_CLASS = {
    "t5-large": T5TODModel,
    "gpt3.5-fs": OpenAIFewShotTODModel,
    "gpt4-fs": OpenAIFewShotTODModel,
}

TOD_MODEL_KWARGS = {
    "t5-large": {
        "model_name_or_path": "t5/output/t5-large-bs32-ep5-olen256/checkpoints",
        "device": "cuda:0",
        "max_context_turns": 0,
        "max_input_length": 512,
        "max_output_length": 256,
        "dst_task_prefix": "対話から信念状態を推定:",
        "rg_task_prefix": "対話から応答を生成:",
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
    },
    "gpt3.5-fs": {
        "openai_model_name": "gpt-3.5-turbo",
        "max_context_turns": 5, # Use 5 context turns on OpenAI model
        "max_output_length": 256,
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
        "faiss_db_fprefix": "llm/output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20",
        "num_fewshot_examples": 2,
    },
    "gpt4-fs": {
        "openai_model_name": "gpt-4-turbo",
        "max_context_turns": 5, # Use 5 context turns on OpenAI model
        "max_output_length": 256,
        "user_utterance_prefix": "<顧客>",
        "system_utterance_prefix": "<店員>",
        "state_prefix": "<信念状態>",
        "db_result_prefix": "<検索結果>",
        "max_candidate_entities": 3,
        "book_result_prefix": "<予約結果>",
        "faiss_db_fprefix": "llm/output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20",
        "num_fewshot_examples": 2,
    }
}

class DialogueGoalSampler:
    def __init__(self, dataset_dpath: str, split: str = "test"):
        self.dataset = JMultiWOZDataset(dataset_dpath=dataset_dpath)
        self.dialogue_names = self.dataset.list_dialogues(split=split)

    def sample(self) -> Tuple[str, dict, str]:
        dialogue_name = random.choice(self.dialogue_names)
        dialogue = self.dataset.get_dialogue(split="test", dialogue_name=dialogue_name)
        return dialogue_name, dialogue["goal"], dialogue["goal_description"]
        

class DialogueSession:
    def __init__(self, session_id: str, dialogue_name: str, goal: dict, goal_description: str,
                 tod_model_name: str, tod_model: TODModelBase, database: JMultiWOZDatabase,
                 max_turns: int, success_phrase: str, failure_phrase: str):
        self.session_id = session_id
        self.dialogue_name = dialogue_name

        self.goal = goal
        self.goal_description = goal_description

        self.tod_model = tod_model
        self.tod_model_name = tod_model_name
        self.tod_model_memory = None

        num_tasks = len(sum(self.goal_description.values(), []))
        self.min_turns = num_tasks * 2 # Assume 2 turns (1 user turn and 1 system turn) per task
        self.max_turns = max_turns
        self.success_phrase = success_phrase
        self.failure_phrase = failure_phrase

        self.database = database

        self.context = []
        self.subjective_success = False
        self.log = []
        
    def check_success_input(self, input_text: str) -> bool:
        return input_text.strip().lower() == self.success_phrase.lower()

    def check_failure_input(self, input_text: str) -> bool:
        return input_text.strip().lower() == self.failure_phrase.lower()
        
    def check_turns_exceeded(self) -> bool:
        return len(self.context) >= self.max_turns

    def model_response(self, input_text: str) -> Tuple[str, str]:
        # Check success
        if self.check_success_input(input_text=input_text):
            if len(self.context) < self.min_turns:
                response = "【対話が短すぎます。もう少し対話を続けてください。】"
                session_over = False
            else:
                response = "【対話を終了します。】"
                session_over = True
                self.subjective_success = True
            return response, session_over
        
        # Check failue
        if self.check_failure_input(input_text=input_text):
            response = "【対話を終了します。】"
            session_over = True
            return response, session_over
        
        # 0. Resume the model's memory and update context
        self.tod_model.set_memory(memory=self.tod_model_memory)
        self.context.append(["USER", input_text])

        # 1. Dialogue State Tracking
        belief_state, book_state = self.tod_model.predict_state(
            context=self.context,
        )

        # 2. Get DB result
        db_result = self.database.get_db_result(
            belief_state=belief_state,
            goal=self.goal,
        )

        # 3. Get Book result
        book_result = self.database.get_book_result(
            book_state=book_state,
            goal=self.goal,
        )

        # 4. Generate response
        response = self.tod_model.generate_response(
            context=self.context,
            belief_state=belief_state,
            book_state=book_state,
            db_result=db_result,
            book_result=book_result,
        )

        # 5. Store the model's memory and update context
        self.tod_model_memory = self.tod_model.get_memory()
        self.context.append(["SYSTEM", response])

        # Check turns
        if self.check_turns_exceeded():
            response = response + "【発話数の上限に達しました。】"
            session_over = True
        else:
            session_over = False

        # Loggging
        self.log.append({
            "input_text": input_text,
            "belief_state": belief_state,
            "book_state": book_state,
            "db_result": db_result,
            "book_result": book_result,
            "response": response,
            "session_over": session_over,
        })

        return response, session_over
    
    def export_to_dict(self) -> dict:
        return deepcopy({
            "session_id": self.session_id,
            "dialogue_name": self.dialogue_name,
            "goal": self.goal,
            "goal_description": self.goal_description,
            "tod_model_name": self.tod_model_name,
            "subjective_success": self.subjective_success,
            "context": self.context,
            "log": self.log,
        })

class JMultiWOZWorld:
    def __init__(self, tod_model_names: List[str], dataset_dpath: str, max_turns: int,
                 success_phrase: str, failure_phrase: str):
        self.tod_models = {}
        for tod_model_name in tod_model_names:
            print(f"Loading {tod_model_name} ...")
            tod_model_class = TOD_MODEL_CLASS[tod_model_name]
            tod_model_kwargs = TOD_MODEL_KWARGS[tod_model_name]
            self.tod_models[tod_model_name] = tod_model_class(**tod_model_kwargs)
        
        self.database = JMultiWOZDatabase(db_dpath=os.path.join(dataset_dpath, "database"))
        self.goal_sampler = DialogueGoalSampler(dataset_dpath=dataset_dpath, split="test")

        self.sessions = {}
        
        self.max_turns = max_turns
        self.success_phrase = success_phrase
        self.failure_phrase = failure_phrase

    def _make_instruction(self, goal_description: dict) -> str:
        instruction = f"""
以下の対話シナリオ文からタスクを読み取り、タスクを達成できるようにチャットボットとの対話を進めてください。<br>
タスクは対話シナリオの上から順番に進めてください。<br>
最大で<b>{self.max_turns}発話</b>（一人あたり<b>{self.max_turns//2}発話</b>）まで対話できます。<br>
全てのタスクを達成できた場合は、 <b>{self.success_phrase}</b> とだけ入力することで、対話を終了できます。<br>
また、対話の継続及びタスク達成が困難な場合は、<b>{self.failure_phrase}</b> と入力して対話を終了できます。<br>
<br>
<strong>対話シナリオ</strong><br>
"""
        desc_list = ""
        for domain_name, domain_description in goal_description.items():
            desc_list += "<li>" + "<br>".join(domain_description) + "</li>"
        
        instruction += "<ul>" + desc_list + "</ul>"
        return instruction

    def create_new_session(self) -> Tuple[str, str]:
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        dialogue_name, goal, goal_description = self.goal_sampler.sample()
        tod_model_name = random.choice(list(self.tod_models))

        self.sessions[session_id] = DialogueSession(
            session_id=session_id,
            dialogue_name=dialogue_name,
            goal=goal,
            goal_description=goal_description,
            tod_model_name=tod_model_name,
            tod_model=self.tod_models[tod_model_name],
            database=self.database,
            max_turns=self.max_turns,
            success_phrase=self.success_phrase,
            failure_phrase=self.failure_phrase,
        )
        instruction = self._make_instruction(goal_description=goal_description)

        return session_id, instruction
    
    def model_response(self, session_id: str, user_input: str) -> Tuple[str, str]:
        session = self.sessions[session_id]

        response_text, session_over = session.model_response(input_text=user_input)

        return response_text, session_over

    def export_session(self, session_id: str, sessions_dpath: str) -> None:
        os.makedirs(sessions_dpath, exist_ok=True)

        result = self.sessions[session_id].export_to_dict()
        json.dump(
            result,
            open(os.path.join(sessions_dpath, f"{session_id}.json"), "w"),
            indent=4, ensure_ascii=False
        )

    def terminate_session(self, session_id: str) -> None:
        del self.sessions[session_id]
