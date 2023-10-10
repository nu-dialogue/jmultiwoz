from typing import List, Tuple

class TODModelBase:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def predict_state(self, context: List[Tuple[str, str]]) -> str:
        """
        Predict belief state from context.
        Args:
            context: List of (speaker, utterance) pairs.
        Returns:
            belief_state: Belief state string.
        """
        raise NotImplementedError
    
    def generate_response(self, context: List[Tuple[str, str]], belief_state: str, db_result: str, book_result: str) -> str:
        """
        Generate system response from context, belief state, db_result, and book_result.
        Args:
            context: List of (speaker, utterance) pairs.
            belief_state: Belief state string.
            db_result: DB result string.
            book_result: Book result string.
        Returns:
            response: System response string.
        """
        raise NotImplementedError
    