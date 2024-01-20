# JMultiWOZ Dataset

## Usage
Unzip the dataset file `JMULtiWOZ_1.0.zip` to get the dataset.
```bash
unzip JMULtiWOZ_1.0.zip
```

You can also use the dataset on [Hugging Face 🤗 Datasets Hub](https://huggingface.co/datasets/nu-dialogue/jmultiwoz) by the following code:
```python
from datasets import load_dataset

dataset = load_dataset("nu-dialogue/jmultiwoz", trust_remote_code=True)
```
See the dataset card for more details.

## Files
```
.
├── database
│   ├── attraction_db.json
│   ├── hotel_db.json
│   ├── restaurant_db.json
│   ├── shopping_db.json
│   ├── taxi_db.json
│   └── weather_db.json
├── dialogues.json
├── informable_slots.json
├── ontology.json
└── split_list.json
```

## Dialogue Data
- The dialogue data is stored in `dialogues.json`. 
- An example of a dialogue is as follows:
    ```json
    {
        ...
        "dialogue_0219prbp": {
            "dialogue_id": 219,
            "dialogue_name": "dialogue_0219prbp",
            "system_name": "operator_0002vBQH",
            "user_name": "customer_0204kXhb",
            "goal": {
                ...
            },
            "goal_description": {
                ...
            },
            "turns": [
                ...
            ]
        },
        ...
    }
    ```
    - `dialogue_id`
        - The dialogue id.
    - `dialogue_name`
        - "dialogue" + (4-digit dialogue id) + (random 4 characters).
    - `system_name`
        - The wizard's worker name.
    - `user_name`
        - The user's worker name.
    - `goal`
        - The goal of the dialogue.
        - The keys are the domain names and the values are the goals of the domains.
        - An example of a dialogue goal is as follows
            ```json
            {
                "general": {
                    "info": {
                        "city": "大阪"
                    }
                },
                "hotel": {
                    "fail_info": {
                        "genre": "旅館",
                        "parking": "有り(無料)",
                        "pricerange": "安め"
                    },
                    "info": {
                        "genre": "dontcare",
                        "parking": "有り(無料)",
                        "pricerange": "安め"
                    },
                    "reqt": [
                        "priceinfo",
                        "name"
                    ]
                },
                ...
            }
            ```
            - `fail_info` (optional)
                - The conditions of the entity that the user fails to find.
            - `info`
                - The conditions of the entity that the user finds.
            - `reqt`
                - The information that the user requests.
            - `fail_book` (optional)
                - The conditions of booking that the user fails to make.
            - `book` (optional)
                - The conditions of booking that the user makes.

    - `goal_description`
        - The description of the goal.
        - The keys are the domain names and the values are the list of the instruction HTML strings.
        - An example of a goal description is as follows:
            ```json
            {
                "general": [
                    "あなたは<b style='color:blue;'>大阪</b>への旅行を計画しています．"
                ],
                "hotel": [
                    "当日泊まる<b style='color:blue;'>宿泊施設</b>を探してください．宿のタイプは<b style='color:blue;'>旅館</b>です．駐車場を<b style='color:blue;'>無料</b>で利用できるところにしてください．予算は<b style='color:blue;'>安め</b>が希望です．",
                    "希望に合うところがない場合は，宿のタイプは<b style='color:blue;'>考慮しない</b>ものとします．",
                    "条件に合う宿泊施設が見つかったら，<b style='color:green;'>料金情報，施設名</b>を聞いてください．"
                ],
                ...
            }
            ```
    - `turns`
        - The list of the dialogue turns.
        - Examples of turns are as follows:
            ```json
            [
                ...
                {
                    "turn_id": 6,
                    "speaker": "USER",
                    "utterance": "ではアスティルホテル十三プレシャスにしようと思うので、料金を教えてください。"
                },
                {
                    "turn_id": 7,
                    "speaker": "SYSTEM",
                    "dialogue_state": {
                        ...
                    },
                    "utterance": "料金情報につきまして、値段帯はお安めとなっておりますが、詳しくは不明です。なお、駐車場は無料でご利用いただけます。"
                },
                ...
            ]
            ```
            - `turn_id`
                - The turn id.
            - `speaker`
                - The speaker of the turn.
            - `dialogue_state` (exists only in the system's turn)
                - The dialogue state.
                - The keys are `belief_state`, `book_state`, `db_result`, and `book_result`.
                - An example of a dialogue state is as follows:
                    ```json
                    {
                        "belief_state": {
                            "general": {
                                "active_domain": "hotel",
                                "city": "大阪"
                            },
                            ...
                            "hotel": {
                                "name": "アスティルホテル十三プレシャス",
                                "genre": null,
                                "area": null,
                                "pricerange": "安め",
                                ...
                            }
                            ...	
                        },
                        "book_state": {
                            "hotel": {
                                "people": null,
                                "day": null,
                                "stay": null
                            },
                            ...	
                        },
                        "db_result": {
                            "candidate_entities": [
                                "アスティルホテル十三プレシャス"
                            ],
                            "active_entity": {
                                "city": "大阪",
                                "name": "アスティルホテル十三プレシャス",
                                "genre": "ビジネスホテル",
                                ...
                            }
                        },
                        "book_result": {
                            "hotel": {
                                "success": null,
                                "ref": null
                            },
                            ...
                        }
                    }
                    ```
                    - `belief_state`
                        - The conditions of the entity that the user seeks.
                    - `book_state`
                        - The conditions of booking that the user seeks.
                    - `db_result`
                        - `candidate_entities`
                            - The list of the entity names that match the user's conditions.
                        - `active_entity`
                            - The entity that the wizard has clicked.
                    - `book_result`
                        - The key is the domain name and the value is the booking result.
                        - `success`
                            - The booking result.
                        - `ref`
                            - The reference number of the booking.
            - `utterance`
                - The utterance of the turn.
                            
## Backend Database
- The backend database is stored in `database/{domain}_db.json`.
- Each file contains the list of the entities of the domain.

> [!NOTE]
> For this databese, we exclusively employed information sources free from copyright constraints. Our list of entities was primarily derived from websites operated by the government or municipalities. Specific information for each entity was solely extracted from their respective official websites, ensuring authenticity and credibility. We consciously abstained from using tourism sites, or any other source potentially encumbered by copyright issues.

> [!WARNING]
> The information in the database may be outdated. Please refer to the official website for the latest information.

- An example of a backend database is as follows:
    ```json
    [  
        {
            "city": "名古屋",
            "name": "名古屋市東山動植物園",
            "genre": [
                "動物園",
                "植物園"
            ],
            "area": "千種区",
            "station": [
                "東山公園駅",
                "星ヶ丘駅"
            ],
            "wifi": "有り(無料)",
            "parking": "有り(有料)",
            "opentime": [
                [
                    "09:00",
                    "16:50"
                ]
            ],
            "phone": "0527822111",
            "address": "愛知県名古屋市千種区東山元町3丁目70",
            "accesstime": [
                [
                    "東山公園駅",
                    "徒歩",
                    "3"
                ],
                [
                    "星ヶ丘駅",
                    "徒歩",
                    "7"
                ]
            ],
            "closed": [
                "月",
                "年末年始"
            ],
            "adultfee": "500",
            "childfee": "0",
            "priceinfo": "区分 大人（高校生以上） 名古屋市在住の65歳以上の方 ／ 観覧券 500円 100円 ／ 団体（有料30名以上） 450円 90円 ／ 団体（有料100名以上） 400円 80円 ／ 年間パスポート（定期観覧券） 2000円 600円 ／ スカイタワー共通券 640円 160円"
        },
        ...
    ]
    ```

## Data Split
- The data split is stored in `split_list.json`.
- The keys are the split names and the values are the list of the dialogue names.
- An example of a data split is as follows:
    ```json
    {
        "train": [
            "dialogue_1704FnBo",
            "dialogue_4687iNha",
            ...
        ],
        "dev": [
            "dialogue_0711tUFS",
            "dialogue_1379EBrk",
            ...
        ],
        "test": [
            "dialogue_0417jMuX",
            "dialogue_3178IBzG",
            ...
        ]
    }
    ```