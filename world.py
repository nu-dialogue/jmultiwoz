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
    "t5-base": T5TODModel,
    "t5-large": T5TODModel,
    "gpt3.5-fs": OpenAIFewShotTODModel,
    "gpt4-fs": OpenAIFewShotTODModel,
}

TOD_MODEL_KWARGS = {
    "t5-base": {
        "model_name_or_path": "t5/output/t5-base-bs32-ep5-olen256/checkpoints",
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
        "openai_model_name": "gpt-4",
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
                 max_turns: int, success_phrase: str, failure_phrase: str,
                 eval_question_list: List[str] = None, eval_answer_list: List[str] = None,):
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

        self.eval_question_list = eval_question_list
        self.eval_answer_list = eval_answer_list
        self.eval_scores = None
        
    def check_success_input(self, input_text: str) -> bool:
        return input_text.strip().lower() == self.success_phrase.lower()

    def check_failure_input(self, input_text: str) -> bool:
        return input_text.strip().lower() == self.failure_phrase.lower()
        
    def check_turns_exceeded(self) -> bool:
        return len(self.context) >= self.max_turns

    def model_response(self, input_text: str) -> Tuple[str, str]:
        # Check success
        if self.check_success_input(input_text=input_text):
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
            "eval_question_list": self.eval_question_list,
            "eval_answer_list": self.eval_answer_list,
            "eval_scores": self.eval_scores,
            "context": self.context,
            "log": self.log,
        })

class JMultiWOZWorld:
    def __init__(self, tod_model_names: List[str], dataset_dpath: str, max_turns: int):
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
        self.success_phrase = "success"
        self.failure_phrase = "fail"

        self.eval_question_list = [
            "1. 全体を通して、チャットボットはあなたの発話を理解できていた。",
            "2. 全体を通して、チャットボットの応答は適切だった。",
            "3. 全体を通して、チャットボットとの対話は満足できるものだった。"
        ]
        self.eval_answer_list = [
            "1. 同意しない", "2. やや同意しない", "3. どちらでもない", "4. やや同意する", "5. 同意する"
        ]

    def _make_instruction(self, goal_description: dict) -> str:
        instruction = f"""
<strong>インストラクション</strong>
<ul>
    <li>以下の<b>対話シナリオ</b>には、今回の対話におけるあなたの設定や目的（いつどこに観光する予定で、ボットに何を案内して欲しいか等）が書かれています。</li>
    <li>このシナリオの順番に対話を進め、ボットから必要な情報を聞き出したり、条件に合った施設をボットに予約してもらったりすることで、<b>シナリオに書かれた目的をすべて達成してください</b>。</li>
    <li>あなたは最大で<b>{self.max_turns//2}回</b>まで発話できます。</li>
    <li>全ての目的を達成できたと思ったら、{self.max_turns//2}発話に達していなくても <b>{self.success_phrase}</b> と入力することで、対話を終了できます。</li>
    <li>また、対話の継続及びタスク達成が困難だと思った場合も、<b>{self.failure_phrase}</b> と入力して対話を終了できます。</li>
</ul>
<strong>注意事項</strong>
<ul>
    <li>目的を達成できたかどうかは、本作業の承認及び報酬の支払いには影響しません。</li>
    <li>本webページを閉じたり更新したりすると、対話がリセットされますのでご注意ください。</li>
    <li>ボットからの応答に時間がかかることがあります（最大1分程度）ので、その際はしばらくお待ちください。</li>
</ul>
<strong>対話シナリオ</strong><br>
"""
        desc_list = ""
        for domain_name, domain_description in goal_description.items():
            desc_list += "<li>" + "<br>".join(domain_description) + "</li>"
        
        instruction += "<ol>" + desc_list + "</ol>"
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
            eval_question_list=self.eval_question_list,
            eval_answer_list=self.eval_answer_list,
        )
        instruction = self._make_instruction(goal_description=goal_description)

        html_format_args = {
            "session_id": session_id,
            "instruction": instruction,
            "question_list": self.eval_question_list,
            "answer_list": self.eval_answer_list,
        }

        return html_format_args
    
    def model_response(self, session_id: str, user_input: str) -> Tuple[str, str]:
        session = self.sessions[session_id]

        response_text, session_over = session.model_response(input_text=user_input)

        return response_text, session_over

    def save_eval_scores(self, session_id: str, eval_scores: List[str]) -> None:
        if session_id not in self.sessions:
            print(f"Session {session_id} does not exist.")
            return
        
        self.sessions[session_id].eval_scores = eval_scores

    def export_session(self, session_id: str, sessions_dpath: str) -> None:
        if session_id not in self.sessions:
            print(f"Session {session_id} does not exist.")
            return
        
        os.makedirs(sessions_dpath, exist_ok=True)

        result = self.sessions[session_id].export_to_dict()
        json.dump(
            result,
            open(os.path.join(sessions_dpath, f"{session_id}.json"), "w"),
            indent=4, ensure_ascii=False
        )

    def terminate_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            print(f"Session {session_id} does not exist.")
            return
        
        del self.sessions[session_id]

    def export_unterminated_sessions(self, sessions_dpath: str) -> None:
        for session_id in list(self.sessions.keys()):
            self.export_session(session_id=session_id, sessions_dpath=sessions_dpath)
            self.terminate_session(session_id=session_id)
