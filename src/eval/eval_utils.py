from typing import List, Dict, Tuple, Any

from src.third_party_utils.nltk_porter_stemmer import PorterStemmer
from src.third_party_utils.spacy_stop_words import STOP_WORDS


def get_label_from_question_metadata(correct_answer_key: str,
                                     question_dict: List[Dict[str, str]]) -> str:
    for item in question_dict:
        try:
            if correct_answer_key == item['label']:
                return item['text']
        except KeyError as exc:
            raise KeyError(f"key='label' or 'text' absent in item:\n{item}.\nException = {exc}")
    raise KeyError(f"key='label' absent in metadata:\n{question_dict}")


def split_question_cause_effect(question: str) -> Tuple[str, str]:
    question = question.lower()
    question_split = question.split("happens, how will it affect")
    cause_part = question_split[0].replace("suppose", "")
    effect_part = question_split[1]
    return cause_part, effect_part


def get_most_similar_idx_word_overlap(p1s: List[str], p2:str):
        if not p1s or not p2:
            return -1
        max_idx = -1
        max_sim = 0.0
        k1 = set(get_content_words(p2))
        for idx, p1 in enumerate(p1s):
            k2 = set(get_content_words(p1))
            sim = 1.0 * len(k1.intersection(k2)) / (1.0 * (len(k1.union(k2))))
            if sim > max_sim:
                max_sim = sim
                max_idx = idx
        return max_idx

def predict_word_overlap_best(input_steps, input_cq, input_eq):
        xsentid = get_most_similar_idx_word_overlap(p1s=input_steps, p2=input_cq)
        ysentid = get_most_similar_idx_word_overlap(p1s=input_steps, p2=input_eq)
        return xsentid, ysentid


def is_stop(cand):
    return cand.lower() in STOP_WORDS

# your other hand => hand
# the hand => hand
def drop_leading_articles_and_stopwords(p):
    # other and another can only appear after the primary articles in first line.
    articles = ["a ", "an ", "the ", "your ", "his ", "their ", "my ", "this ", "that ",
                "another ", "other ", "more ", "less "]
    for article in articles:
        if p.lower().startswith(article):
            p = p[len(article):]
    words = p.split(" ")
    answer = ""
    for idx, w in enumerate(words):
        if is_stop(w):
            continue
        else:
            answer = " ".join(words[idx:])
            break
    return answer


def stem(w: str):
    if not w or len(w.strip()) == 0:
        return ""
    w_lower = w.lower()
    # Remove leading articles from the phrase (e.g., the rays => rays).
    # FIXME: change this logic to accept a list of leading articles.
    if w_lower.startswith("a "):
        w_lower = w_lower[2:]
    elif w_lower.startswith("an "):
        w_lower = w_lower[3:]
    elif w_lower.startswith("the "):
        w_lower = w_lower[4:]
    elif w_lower.startswith("your "):
        w_lower = w_lower[5:]
    elif w_lower.startswith("his "):
        w_lower = w_lower[4:]
    elif w_lower.startswith("their "):
        w_lower = w_lower[6:]
    elif w_lower.startswith("my "):
        w_lower = w_lower[3:]
    elif w_lower.startswith("another "):
        w_lower = w_lower[8:]
    elif w_lower.startswith("other "):
        w_lower = w_lower[6:]
    elif w_lower.startswith("this "):
        w_lower = w_lower[5:]
    elif w_lower.startswith("that "):
        w_lower = w_lower[5:]
    # Porter stemmer: rays => ray
    return PorterStemmer().stem(w_lower).strip()


def stem_words(words):
    return [stem(w) for w in words]


def get_content_words(s):
    para_words_prev = s.strip().lower().split(" ")
    para_words = set()
    for word in para_words_prev:
        if not is_stop(word):
            para_words.add(word)
    return stem_words(list(para_words))


def find_max(input_list: List[float]) -> Tuple[Any, int]:
    max_item = max(input_list)
    return max_item, input_list.index(max_item)
