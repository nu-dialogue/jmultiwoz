from typing import List, Tuple, Optional

from utils.data_utils import (
    context_list2str,
    domain_state_dict2str,
    db_result_dict2str,
    domain_book_result_dict2str,
)

STATE_TRACKING_PROMPTS = {
"general": """
対話文脈から，現時点での話題に関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "active_domain" 最後に顧客が言及した最新の話題．"restaurant/hotel/attraction/shopping/taxi/weather" のいずれか．
- "city" 旅行先の都市．"札幌/仙台/東京/横浜/名古屋/京都/大阪/福岡/那覇" のいずれか．
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

# === 対話例 1 ===
# ### 入力
# <顧客> 福岡へ行くよていなのですが、値段が普通くらいの宿泊施設を探してもらっていいですか？ <店員> かしこまりました。ではWITH THE STYLE FUKUOKAはいかがでしょうか。 <顧客> "なるほど。でも最寄り駅が海ノ中道駅で、駐車場を無料で利用できるとこがいいんですよね。
# ### 出力
# <信念状態> active_domain hotel, city 福岡
# === 対話例 2 ===
# ### 入力
# <顧客> そうですか。では、スポーツ用品以外でも大丈夫です。 <店員> 大阪中央区には、駐車場無料の買い物施設は３件ありますが、なんばCITYをお勧めします。 <顧客> そこにします。住所と営業時間と近辺の駅からの所要時間を調べてもらえます？ <店員> 住所は大阪府大阪市中央区難波5丁目1-60、営業時間は11:00～21:00、難波駅から徒歩で1分です。 <顧客> わかりました。Wi-Fiと駐車場が無料利用できる観光名所も探してほしいです。
# ### 出力
# <信念状態> active_domain attraction, city 大阪
# === 対話例 3 ===
# ### 入力
# <店員> エリアでいうと太白区にある、仙台市野草園はいかがでしょう？ジャンルは、自然、公園となります。 <顧客> そこにします。定休日はありますか？ <店員> 定休日は12月1日から3月19日となります。 <顧客> では、9月30日の青葉区の天気はわかりますか？
# ### 出力
# <信念状態> active_domain weather, city 仙台
# ======
# では，以下の事例を完了してください:

"restaurant": """
対話文脈から判断できる，レストランに関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "name" レストランの名前
- "genre" レストランのジャンル
- "area" レストランのエリア．"~区" や "~市" などの地区名．
- "pricerange" レストランの価格帯．"安め/普通/高め" のいずれか．
- "station" レストランの最寄り駅
- "wifi" レストランのWi-Fiの有無．"有り(無料)/有り(有料)/無し" のいずれか．
- "parking" レストランの駐車場の有無．"有り(無料)/有り(有料)/無し" のいずれか．
- "people" レストランの予約人数
- "day" レストランの予約日
- "time" レストランの予約時間
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

"hotel": """
対話文脈から判断できる，ホテルに関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "name" 宿泊施設の名前
- "genre" 宿泊施設のジャンル
- "area" 宿泊施設のエリア．"~区" や "~市" などの地区名．
- "pricerange" 宿泊施設の価格帯．"安め/普通/高め" のいずれか．
- "station" 宿泊施設の最寄り駅
- "wifi" 宿泊施設のWi-Fiの有無．"有り(無料)/有り(有料)/無し" のいずれか．
- "parking" 宿泊施設の駐車場の有無．"有り(無料)/有り(有料)/無し" のいずれか．
- "withrestaurant" 宿泊施設にレストランがあるかどうか．"有り/無し" のいずれか．
- "people" 宿泊施設の予約人数
- "day" 宿泊施設の予約日
- "stay" 宿泊施設の宿泊日数
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

"attraction": """
対話文脈から判断できる，観光名所に関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "name" 観光名所の名前
- "genre" 観光名所のジャンル
- "area" 観光名所のエリア．"~区" や "~市" などの地区名．
- "station" 観光名所の最寄り駅
- "wifi" 観光名所のWi-Fiの有無．"有り(無料)/有り(有料)/無し" のいずれか．
- "parking" 観光名所の駐車場の有無．"有り(無料)/有り(有料)/無し" のいずれか．
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

"shopping": """
対話文脈から判断できる，ショッピング施設に関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "name" ショッピング施設の名前
- "genre" ショッピング施設のジャンル
- "area" ショッピング施設のエリア．"~区" や "~市" などの地区名．
- "station" ショッピング施設の最寄り駅
- "parking" ショッピング施設の駐車場の有無．"有り(無料)/有り(有料)/無し" のいずれか．
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

"taxi": """
対話文脈から判断できる，タクシーに関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "name" タクシー会社の名前
- "cashless" タクシー会社のキャッシュレス決済の有無．"対応/非対応" のいずれか．
- "jumbo" タクシー会社のジャンボタクシーの有無．"対応/非対応" のいずれか．
- "day" タクシーの予約日
- "time" タクシーの予約時間
- "departurepoint" タクシーの出発地点
- "arrivalpoint" タクシーの到着地点
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
""",

"weather": """
対話文脈から判断できる，天気に関するスロットと値ペア（信念状態）を抽出してください．
信念状態は半角スペースとカンマを使用し，`スロット1 値1, スロット2 値2` という形式で抽出してください．
抽出するべきスロットは以下の通りです：
- "area" 天気の地域．"~区" や "~市" などの地区名．
- "day" 天気の日付
上記以外の情報は抽出しないでください．
また，文脈中で言及されなかったスロットの値も抽出しないでください．
"""
}

RESPONSE_GENERATION_PROMPTS = {
"general": """"
あなたは，顧客の旅行計画をサポートし，情報案内をするアシスタントです．
データベースを用い，レストラン，ホテル，観光地，ショッピング，タクシー，天気予報などの情報を提供することができます．
文脈に沿って応答し，顧客から尋ねられた情報を提供してください．
""",

"restaurant": """
あなたは顧客の要望に沿ったレストランを探し出し，予約をするアシスタントです．
データベースを用い，エリア，ジャンル，価格帯等からレストランを検索・予約することができます．
レストランを見つけたら，その名前，住所，電話番号，その他必要な情報など，顧客から尋ねられた情報を提供してください．
予約が成功したら，その予約番号（ref）を提供してください
""",

"hotel": """
あなたは顧客の要望に沿った宿泊施設を探し出し，予約をするアシスタントです．
データベースを用い，エリア，ジャンル，価格帯等から宿泊施設を検索・予約することができます．
宿泊施設を見つけたら，その名前，住所，電話番号，その他必要な情報など，顧客から尋ねられた情報を提供してください．
予約が成功したら，その予約番号（ref）を提供してください
""",

"attraction": """
あなたは顧客の要望に沿った観光名所を探し出し，情報を提供するアシスタントです．
データベースを用い，エリア，ジャンル等から観光名所を検索することができます．
観光名所を見つけたら，その名前，住所，電話番号，その他必要な情報など，顧客から尋ねられた情報を提供してください．
""",

"shopping": """
あなたは顧客の要望に沿ったショッピング施設を探し出し，情報を提供するアシスタントです．
データベースを用い，エリア，ジャンル等からショッピング施設を検索することができます．
ショッピング施設を見つけたら，その名前，住所，電話番号，その他必要な情報など，顧客から尋ねられた情報を提供してください．
""",

"taxi": """
あなたは顧客の要望に沿ったタクシー会社を探し出し，予約をするアシスタントです．
データベースを用い，キャッシュレス決済，ジャンボタクシー等からタクシー会社を検索・予約することができます．
タクシー会社を見つけたら，その名前，住所，電話番号，その他必要な情報など，顧客から尋ねられた情報を提供してください．
予約が成功したら，その予約番号（ref）を提供してください
""",

"weather": """
あなたは顧客の要望に沿った天気予報を探し出し，情報を提供するアシスタントです．
データベースを用い，地域，日付等から天気予報を検索することができます．
天気予報を見つけたら，その日の天気，最高気温，最低気温，降水確率，その他必要な情報など，顧客から尋ねられた情報を提供してください．
"""
}

class PromptFormater:
    def __init__(self, max_context_turns: int, user_utterance_prefix: str, system_utterance_prefix: str,
                 state_prefix: str, db_result_prefix: str, max_candidate_entities: int, book_result_prefix: str):
        self.max_context_turns = max_context_turns
        self.user_utterance_prefix = user_utterance_prefix
        self.system_utterance_prefix = system_utterance_prefix
        self.state_prefix = state_prefix
        self.db_result_prefix = db_result_prefix
        self.max_candidate_entities = max_candidate_entities
        self.book_result_prefix = book_result_prefix
        
    def make_state_prompt(self, domain: str, context: List[Tuple[str, str]], fewshot_examples: Optional[List[dict]] = None):
        prompt = STATE_TRACKING_PROMPTS[domain]

        if fewshot_examples:
            for i, example in enumerate(fewshot_examples):
                context_str = context_list2str(
                    context=example["context"],
                    max_context_turns=self.max_context_turns,
                    user_utterance_prefix=self.user_utterance_prefix,
                    system_utterance_prefix=self.system_utterance_prefix
                )
                state_str = domain_state_dict2str(
                    domain=domain,
                    belief_state=example["belief_state"],
                    book_state=example["book_state"]
                )
                prompt += (f"\n=== 対話例 {i+1} ===\n"
                           f"### 入力\n"
                           f"{context_str}\n"
                           f"### 出力\n"
                           f"{self.state_prefix} {state_str}\n")
            prompt += ("======\n"
                       "では，以下の事例を完了してください:\n")
        
        context_str = context_list2str(
            context=context,
            max_context_turns=self.max_context_turns,
            user_utterance_prefix=self.user_utterance_prefix,
            system_utterance_prefix=self.system_utterance_prefix
        )
        prompt += (f"### 入力\n"
                   f"{context_str}\n"
                   f"### 出力\n"
                   f"{self.state_prefix} ")
        
        return prompt

    def make_response_prompt(self, domain: str, context: List[Tuple[str, str]], belief_state: dict, book_state: dict,
                             db_result: dict, book_result: dict, fewshot_examples: Optional[List[dict]] = None):
        prompt = RESPONSE_GENERATION_PROMPTS[domain]

        if fewshot_examples:
            for i, example in enumerate(fewshot_examples):
                context_str = context_list2str(
                    context=example["context"],
                    max_context_turns=self.max_context_turns,
                    user_utterance_prefix=self.user_utterance_prefix,
                    system_utterance_prefix=self.system_utterance_prefix
                )
                state_str = domain_state_dict2str(
                    domain=domain,
                    belief_state=example["belief_state"],
                    book_state=example["book_state"]
                )
                db_result_str = db_result_dict2str(
                    db_result=example["db_result"],
                    max_candidate_entities=self.max_candidate_entities
                )
                book_result_str = domain_book_result_dict2str(
                    domain=domain,
                    book_result=example["book_result"]
                )
                response = example["response"]
                prompt += (f"\n=== 対話例 {i+1} ===\n"
                            "### 入力\n"
                           f"{context_str}\n"
                           f"{self.state_prefix} {state_str}\n"
                           f"{self.db_result_prefix} {db_result_str}\n"
                           f"{self.book_result_prefix} {book_result_str}\n"
                            "### 出力\n"
                           f"{self.system_utterance_prefix} {response}\n")
            prompt += ("======\n"
                       "では，以下の事例を完了してください:\n")
        
        context_str = context_list2str(
            context=context,
            max_context_turns=self.max_context_turns,
            user_utterance_prefix=self.user_utterance_prefix,
            system_utterance_prefix=self.system_utterance_prefix
        )
        state_str = domain_state_dict2str(
            domain=domain,
            belief_state=belief_state,
            book_state=book_state
        )
        db_result_str = db_result_dict2str(
            db_result=db_result,
            max_candidate_entities=self.max_candidate_entities
        )
        book_result_str = domain_book_result_dict2str(
            domain=domain,
            book_result=book_result
        )

        prompt += (f"### 入力\n"
                   f"{context_str}\n"
                   f"{self.state_prefix} {state_str}\n"
                   f"{self.db_result_prefix} {db_result_str}\n"
                   f"{self.book_result_prefix} {book_result_str}\n"
                   f"### 出力\n"
                   f"{self.system_utterance_prefix} ")
        
        return prompt
