import os
import json
import pandas as pd
import random
from typing import List, Tuple
from copy import deepcopy

class JMultiWOZDataset:
    def __init__(self, dataset_dpath) -> None:
        self.dataset_dpath = dataset_dpath
        self.ontology = json.load(open(os.path.join(dataset_dpath, "ontology.json")))
        self.split_list = json.load(open(os.path.join(dataset_dpath, "split_list.json")))
        dialogues = json.load(open(os.path.join(dataset_dpath, "dialogues.json")))

        self.dialogues = {}
        for split, dialogue_names in self.split_list.items():
            self.dialogues[split] = {}
            for dialogue_name in dialogue_names:
                self.dialogues[split][dialogue_name] = dialogues[dialogue_name]
    
    @property
    def available_domains(self) -> List[str]:
        return list(self.ontology.keys())

    def list_dialogues(self, split: str) -> List[str]:
        return self.split_list[split]
    
    def get_dialogues(self, split: str) -> dict:
        return self.dialogues[split]

    def get_dialogue(self, split: str, dialogue_name: str) -> dict:
        return self.dialogues[split][dialogue_name]

    def iter_dialogue_turns(self, split: str, dialogue_name: str) -> Tuple[List[Tuple[str, str]], dict]:
        dialogue = self.dialogues[split][dialogue_name]
        context = []
        for turn in dialogue["turns"]:
            if turn["speaker"] == "USER":
                context = context + [("USER", turn["utterance"])]
            else:
                yield context, turn
                context = context + [("SYSTEM", turn["utterance"])]

class JMultiWOZDatabase:
    def __init__(self, db_dpath):
        self.db_dfs = {}
        domain_names = ["attraction", "hotel", "restaurant", "shopping", "taxi", "weather"]
        for domain_name in domain_names:
            domain_db_dicts = json.load(open(os.path.join(db_dpath, f"{domain_name}_db.json")))
            domain_db_df = pd.DataFrame(domain_db_dicts)
            domain_db_df["domain"] = domain_name
            self.db_dfs[domain_name] = domain_db_df
            
        self.not_query_slots = []
        self.not_query_values = [None, "", "dontcare", "非対応", "無し", "不明"]
        
    def query(self, domain_name: str, city_name: str, constraints: dict):
        df = self.db_dfs[domain_name].copy()
        
        constraints = {"city": city_name, **constraints}

        for slot_name, value in constraints.items():
            if slot_name in self.not_query_slots:
                continue
            if value in self.not_query_values:
                continue

            def is_equal(value_const, value_db):
                if not isinstance(value_const, list) and not isinstance(value_db, list):
                    return value_const == value_db
                elif not isinstance(value_const, list) and isinstance(value_db, list):
                    return value_const in value_db
                else:
                    raise ValueError(f"not implemented: {value_const}, {value_db}")

            df = df[df[slot_name].map(lambda value_db: is_equal(value, value_db)).astype(bool)]
        
        return df

    def get_entity_names(self, domain_name, city_name, constraints, negative_constraints):
        try:
            entities = self.query(domain_name=domain_name, city_name=city_name, constraints=constraints)
        except Exception as e:
            print(domain_name, constraints)
            raise e

        if negative_constraints:
            try:
                negative_entities = self.query(domain_name=domain_name, city_name=city_name, constraints=negative_constraints)
            except Exception as e:
                print(domain_name, negative_constraints)
                raise e
            entities = entities.loc[~entities.index.isin(negative_entities.index)]
        
        return entities["name"].tolist()

    def get_entity_info(self, domain_name, entity_name, city_name):
        entities = self.query(domain_name=domain_name,
                              city_name=city_name,
                              constraints={"name": entity_name})
        if entities.shape[0] == 0:
            entities = self.query(domain_name=domain_name,
                                  city_name=None,
                                  constraints={"name": entity_name})
            
        assert entities.shape[0] > 0, f"{domain_name} / {entity_name} is not found"
        
        return entities.to_dict(orient="records")[0]
    
    def get_db_result(self, belief_state, goal, oracle_db_result=None):
        active_domain = belief_state["general"]["active_domain"]
        city_name = belief_state["general"]["city"]

        if active_domain is None:
            candidate_entities = []
        else:
            fail_info = goal.get(active_domain, {}).get("fail_info", {})
            candidate_entities = self.get_entity_names(
                domain_name=active_domain,
                city_name=city_name,
                constraints=belief_state[active_domain],
                negative_constraints=fail_info
            )
            if active_domain == "weather" and len(candidate_entities) > 1:
                candidate_entities = [] # Remove weather entities because of the too many candidates

        if len(candidate_entities) == 0:
            active_entity = None
        elif len(candidate_entities) >= 1:
            if oracle_db_result and oracle_db_result["active_entity"] and oracle_db_result["active_entity"]["name"] in candidate_entities:
                entity_name = oracle_db_result["active_entity"]["name"]
            else:
                entity_name = candidate_entities[0]
            active_entity = self.get_entity_info(
                domain_name=active_domain,
                city_name=city_name,
                entity_name=entity_name
            )
        else:
            active_entity = None

        db_result = {
            "candidate_entities": candidate_entities,
            "active_entity": active_entity,
        }
        return db_result
    
    def get_book_result(self, book_state, goal, oracle_book_result=None):
        book_result = {}
        for domain, constraints in book_state.items():
            book_result[domain] = {"success": None, "ref": None}
            
            constraints = deepcopy(constraints)
            true_constraints = deepcopy(goal.get(domain, {}).get("book", {}))

            if all(constraints.values()):
                # Remove some slots from taxi book query
                if domain == "taxi":
                    # We don't use the following taxi slots since their true
                    # values change dynamically during dialogues.
                    taxi_slots_to_remove = ["departurepoint", "arrivalpoint"]
                    for slot_name in taxi_slots_to_remove:
                        del constraints[slot_name], true_constraints[slot_name]

                # Check success or not
                if constraints == true_constraints: # Success
                    book_result[domain]["success"] = True
                    # Use oracle ref if exists
                    if oracle_book_result and oracle_book_result.get(domain, {}).get("ref"):
                        book_result[domain]["ref"] = oracle_book_result[domain]["ref"] # Use oracle ref
                    else:
                        book_result[domain]["ref"] = random.randrange(1, 10**5) # Random ref

                else: # Fail
                    book_result[domain]["success"] = False

        return book_result
