import torch
from transformers import BertTokenizer, BertModel
from nltk.metrics import edit_distance

# **1. Объяснение UD-тегов синтаксических отношений:**
ud_tags_example = {
    "nsubj": "В предложении 'Кот ловит мышь' тег 'кот' - это nsubj, выполняющий действие.",
    "dobj": "В предложении 'Кот ловит мышь' тег 'мышь' - это dobj, объект действия.",
    "amod": "В предложении 'Большой кот ловит мышь' тег 'большой' - это amod, описание существительного."
}

for tag, explanation in ud_tags_example.items():
    print(f"{tag}: {explanation}")


# **2. Функция разбиения сложносочиненного предложения на простые:**
def split_compound_sentence(sentence):
    parts = sentence.split(',')
    return [part.strip() for part in parts]


# Пример использования
compound_sentence = "Я люблю читать, а мой друг предпочитает смотреть фильмы"
simple_sentences = split_compound_sentence(compound_sentence)
print("\nРазбиение сложносочиненного предложения на простые:")
for idx, sent in enumerate(simple_sentences, 1):
    print(f"Предложение {idx}: {sent}")


# **3. Функция нахождения наименьшего общего предка двух токенов:**
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def find_lca(root, token1, token2):
    if root is None:
        return None

    if root.value == token1 or root.value == token2:
        return root

    found = [child for child in root.children if find_lca(child, token1, token2) is not None]

    if len(found) == 2:
        return root
    return found[0] if found else None


# Пример создания дерева и нахождения LCA
root = TreeNode("S")
child1 = TreeNode("VP")
child2 = TreeNode("NP")
root.children.append(child1)
root.children.append(child2)
child1.children.append(TreeNode("V"))
child1.children.append(TreeNode("NP"))
child2.children.append(TreeNode("N"))
child2.children.append(TreeNode("Adj"))

lca = find_lca(root, "V", "N")
print("\nНаименьший общий предок токенов 'V' и 'N':", lca.value if lca else None)


# **4. Сравнение предложений двумя методами:**
def compare_sentences(sentence1, sentence2):
    # Метод 1: Расстояние редактирования
    edit_dist = edit_distance(sentence1, sentence2)

    # Метод 2: Косинусная мера БERT-эмбеддингов
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs1 = tokenizer(sentence1, return_tensors='pt')
    inputs2 = tokenizer(sentence2, return_tensors='pt')

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

    return edit_dist, cos_sim.item()

# Пример использования
sentence_a = "Кот ловит мышь"
sentence_b = "Кот поймает мышь"
results = compare_sentences(sentence_a, sentence_b)
print("\nСравнение предложений:")
print("Редактирование:", results[0], "Косинусное сходство:", results[1])
