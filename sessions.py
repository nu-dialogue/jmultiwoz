import os
import openai
import random, string

openai.api_key = os.environ.get('OPENAI_API_KEY')

class Session:
    def __init__(self, goal: dict, goal_description: str, tod_model_name: str,
                 tod_model: TODModelBase, database: JMultiWOZDatabase):
        self.tod_model = tod_model

        self.goal = goal
        self.goal_description = goal_description
        self.tod_model_name = tod_model_name
        self.tod_model_memory = None
        self.context = []

        self.database = database

    def model_response(self, input_text):
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

        return response

class JMultiWOZSessions:
    def __init__(self, tod_models: Dict[str, TODModelBase], dataset_dpath: str):
        self.tod_models = tod_models
        self.database = ...
        self.goal_sampler = ...

        self.sessions = {}

    def make_new_session(self):
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        goal, goal_description = self.goal_sampler.sample()
        tod_model_name = random.choice(list(self.tod_models))

        self.sessions[session_id] = Session(
            goal=goal,
            goal_description=goal_description,
            tod_model_name=tod_model_name,
            tod_model=self.tod_models[tod_model_name],
            database=self.database,
        )
        goal_desc_str = "\n".join([sum(goal_description.values(), [])])

        return session_id, goal_desc_str
    
    def response(self, session_id: str, user_input: str):
        session = self.sessions[session_id]
        return session.model_response(input_text=user_input)

