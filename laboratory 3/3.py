import time
import numpy as np
from typing import List, Dict
from typing import Tuple
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity
from pyaspeller import YandexSpeller
from textblob import TextBlob
import nltk
import random
from dataclasses import dataclass

import pandas as pd

nltk.download('punkt')
test_texts = [ "вот в инете откапал такую n интеерсную статейку",
    "предлагаю вашему внимани",
    "может и в_правду лутше тебе молчать",
    "утром мы сидели как сычи",
    "превет всем",
    "это тилифон"]


def measure_time(func, text: str, iterations: int = 3) -> float:
    times = []
    for _ in range(iterations):
        start_time = time.time()
        _ = func(text)
        times.append(time.time() - start_time)
    return np.mean(times)
from itertools import islice

import pkg_resources
from symspellpy import SymSpell

sym_spell = SymSpell()
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1)

# Print out first 5 elements to demonstrate that dictionary is
# successfully loaded
print(list(islice(sym_spell.words.items(), 5)))



def symspell_correct(text: str) -> str:
    words = text.split()
    corrected_words = []

    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)

corrector_yandex = YandexSpeller()
def yandex_correct(text: str) -> str:
    return corrector_yandex.spelled_text(text)

def textblob_correct(text: str) -> str:
    blob = TextBlob(text)
    return str(blob.correct())

results = {}
methods = {'SymSpell': symspell_correct,
    'YandexSpeller': yandex_correct,
    'TextBlob': textblob_correct}

for name, method in methods.items():
    method_times = []
    print(f"\nтест {name}:")

    for text in test_texts:
        try:
            avg_time = measure_time(method, text)
            method_times.append(avg_time)

            # коррекции и время
            corrected = method(text)
            print(f"текст: {text}")
            print(f"исправленный: {corrected}")
            print(f"время: {avg_time:.4f} сек\n")
        except Exception as e:
            print(f"ошибка '{text}': {str(e)}")
            continue

    if method_times:
        results[name] = {
            'avg_time': np.mean(method_times),
            'std_time': np.std(method_times),
            'min_time': min(method_times),
            'max_time': max(method_times)
        }