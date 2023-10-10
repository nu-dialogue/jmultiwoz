from copy import deepcopy
from typing import List, Tuple
import warnings
import random

default_belief_state = {
    "general": {
        "active_domain": None,
        "city": None,
    },
    "restaurant": {
        "name": None,
        "genre": None,
        "area": None,
        "pricerange": None,
        "station": None,
        "wifi": None,
        "parking": None,
    },
    "hotel": {
        "name": None,
        "genre": None,
        "area": None,
        "pricerange": None,
        "station": None,
        "wifi": None,
        "parking": None,
        "withrestaurant": None,
    },
    "attraction": {
        "name": None,
        "genre": None,
        "area": None,
        "station": None,
        "wifi": None,
        "parking": None
    },
    "shopping": {
        "name": None,
        "genre": None,
        "area": None,
        "station": None,
        "parking": None
    },
    "taxi": {
        "name": None,
        "cashless": None,
        "jumbo": None,
    },
    "weather": {
        "area": None,
        "day": None,
    }   
}

default_book_state = {
    "restaurant": {
        "people": None,
        "day": None,
        "time": None,
    },
    "hotel": {
        "people": None,
        "day": None,
        "stay": None,
    },
    "taxi": {
        "day": None,
        "time": None,
        "departurepoint": None,
        "arrivalpoint": None
    }
}

default_book_result = {
    "restaurant": {
        "success": None,
        "ref": None,
    },
    "hotel": {
        "success": None,
        "ref": None,
    },
    "taxi": {
        "success": None,
        "ref": None,
    }
}

def state_dict2str(belief_state: dict, book_state: dict) -> str:
    flat_belief_state = []
    for domain, slot_values in belief_state.items():
        for slot, value in slot_values.items():
            if not value:
                continue
            flat_belief_state.append(f"{domain} {slot} {value}")
        for slot, value in book_state.get(domain, {}).items():
            if not value:
                continue
            flat_belief_state.append(f"{domain} {slot} {value}")
    return ", ".join(flat_belief_state)

def state_str2dict(state_str: str) -> Tuple[dict, dict]:
    belief_state = deepcopy(default_belief_state)
    book_state = deepcopy(default_book_state)

    for slot_value in state_str.split(", "):
        try:
            domain, slot, value = slot_value.split(" ", 2)
        except ValueError:
            warnings.warn(f"Invalid slot_value: {slot_value}")
            continue
        if slot in belief_state.get(domain, {}):
            belief_state[domain][slot] = value
        elif slot in book_state.get(domain, {}):
            book_state[domain][slot] = value
        else:
            warnings.warn(f"Unknown slot: {domain} {slot} {value}")
    return belief_state, book_state

def db_result_dict2str(db_result, max_candidate_entities):
    candidate_entities = deepcopy(db_result["candidate_entities"])

    num_candidate_entities = len(candidate_entities)

    truncated_candidate_entities = []
    if candidate_entities and db_result["active_entity"]:
        active_entity_index = candidate_entities.index(db_result["active_entity"]["name"])
        truncated_candidate_entities.append(candidate_entities.pop(active_entity_index))
        max_candidate_entities = max_candidate_entities - 1

    truncated_candidate_entities += random.sample(
        candidate_entities, min(max_candidate_entities, len(candidate_entities))
    )
    candidate_line = "[" + ", ".join(truncated_candidate_entities) + "]"

    def _flatten_value(values):
        if not isinstance(values, list):
            return values
        flat_values = [
            _flatten_value(v) if isinstance(v, list) else v for v in values
        ]
        return "[" + ", ".join(flat_values) + "]"
    
    active_entity_info = []
    if db_result["active_entity"]:
        for slot, value in db_result["active_entity"].items():
            value = _flatten_value(value)
            active_entity_info.append(f"{slot} {value}")
    active_entity_line = "[" + ", ".join(active_entity_info) + "]"

    return (f"N {num_candidate_entities}, "
            f"candidate {candidate_line}, "
            f"selected {active_entity_line}")

def book_result_dict2str(book_result):
    flat_book_result = []
    for domain, slot_values in book_result.items():
        for slot, value in slot_values.items():
            if not value:
                continue
            flat_book_result.append(f"{domain} {slot} {value}")
    return ", ".join(flat_book_result)

def book_result_str2dict(book_result_str):
    book_result = deepcopy(default_book_result)
    for slot_value in book_result_str.split(", "):
        domain, slot, value = slot_value.split(" ", 2)
        if slot == "success":
            value = True
        book_result[domain][slot] = value
    return book_result