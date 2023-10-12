from typing import List, Tuple

class TODModelBase:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    def init_session(self):
        """
        Initialize dialogue session.
        """
        raise NotImplementedError

    def predict_state(self, context: List[Tuple[str, str]]) -> Tuple[dict, dict]:
        """
        Predict belief state from context.
        Args:
            context: List of (speaker, utterance) pairs.
        Returns:
            belief_state: Belief state.
        """
        raise NotImplementedError
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: dict, book_state: dict,
                          db_result: dict, book_result: dict) -> str:
        """
        Generate system response from context, belief state, db_result, and book_result.
        Args:
            context: List of (speaker, utterance) pairs.
            belief_state: Belief state.
            db_result: DB result.
            book_result: Book result.
        Returns:
            response: System response.
        """
        raise NotImplementedError
    