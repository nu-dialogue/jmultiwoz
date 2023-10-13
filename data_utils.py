import os
import json
from copy import deepcopy
from typing import List, Tuple
import warnings
import random
from tqdm import tqdm

informable_slots = json.load(open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset/JMultiWOZ_1.0/informable_slots.json"
    )
))

informable_values = {}
for slot_name, values in informable_slots.items():
    informable_values[slot_name] = [v["value"] for v in values]

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

def context_list2str(context: List[Tuple[str, str]], max_context_turns: int,
                     user_utterance_prefix: str, system_utterance_prefix: str) -> str:
    speaker2prefix = {"USER": user_utterance_prefix,
                      "SYSTEM": system_utterance_prefix}
    context_str = " ".join([f"{speaker2prefix[speaker]} {utterance}"
                            for speaker, utterance in context[-max_context_turns:]])
    return context_str

def state_dict2str(belief_state: dict, book_state: dict) -> str:
    flat_state = []
    for domain, slot_values in belief_state.items():
        for slot, value in slot_values.items():
            if not value:
                continue
            flat_state.append(f"{domain} {slot} {value}")
        for slot, value in book_state.get(domain, {}).items():
            if not value:
                continue
            flat_state.append(f"{domain} {slot} {value}")
    return ", ".join(flat_state)

def state_str2dict(state_str: str) -> Tuple[dict, dict]:
    belief_state = deepcopy(default_belief_state)
    book_state = deepcopy(default_book_state)

    for slot_value in state_str.split(", "):
        try:
            domain, slot, value = slot_value.split(" ", 2)
        except ValueError:
            warnings.warn(f"Invalid slot_value: {slot_value}")
            continue
        if slot in informable_slots and not value in informable_values[slot]:
            warnings.warn(f"Unknown slot: {domain} {slot} {value}")
            continue

        if slot in belief_state.get(domain, {}):
            belief_state[domain][slot] = value
        elif slot in book_state.get(domain, {}):
            book_state[domain][slot] = value
        else:
            warnings.warn(f"Unknown slot: {domain} {slot} {value}")
    return belief_state, book_state

def domain_state_dict2str(domain: str, belief_state: dict, book_state: dict):
    flat_state = []
    for slot, value in belief_state[domain].items():
        if not value:
            continue
        flat_state.append(f"{slot} {value}")
    for slot, value in book_state.get(domain, {}).items():
        if not value:
            continue
        flat_state.append(f"{slot} {value}")
    return ", ".join(flat_state)

def domain_state_str2dict(domain: str, domain_state_str: str):
    domain_belief_state = deepcopy(default_belief_state)[domain]
    domain_book_state = deepcopy(default_book_state).get(domain, {})

    if not domain_state_str:
        return domain_belief_state, domain_book_state
    
    for slot_value in domain_state_str.split(", "):
        try:
            slot, value = slot_value.split(" ", 1)
        except ValueError:
            warnings.warn(f"Invalid slot_value: {slot_value}")
            continue
        if slot in informable_slots and value not in informable_values[slot]:
            warnings.warn(f"Unknown slot: {domain} {slot} {value}")
            continue

        if slot in domain_belief_state:
            domain_belief_state[slot] = value
        elif slot in domain_book_state:
            domain_book_state[slot] = value
        else:
            warnings.warn(f"Unknown slot: {domain} {slot} {value}")
    return domain_belief_state, domain_book_state

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

    return (f"total {num_candidate_entities}, "
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

def domain_book_result_dict2str(domain, book_result):
    flat_book_result = []
    for slot, value in book_result.get(domain, {}).items():
        if not value:
            continue
        flat_book_result.append(f"{slot} {value}")
    return ", ".join(flat_book_result)
