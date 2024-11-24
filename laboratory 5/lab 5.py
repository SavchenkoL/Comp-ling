import spacy
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel

#1. Объяснить значение трех UD-тегов синтаксических отношений на примере фрагмента размеченного корпуса
def parse_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        parsed_data = []

    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue

        columns = line.strip().split('\t')

        if len(columns) == 10:
            token_id, word, lemma, upos, xpos, feats, head, dep_rel, _, _ = columns
            parsed_data.append({
                'token_id': token_id,
                'word': word,
                'lemma': lemma,
                'upos': upos,
                'xpos': xpos,
                'head': head,
                'dep_rel': dep_rel
            })

    return parsed_data

def analyze_syntactic_relations(parsed_data):
    relations = {'nsubj': [], 'root': [], 'dobj': []}

    for token in parsed_data:
        dep_rel = token['dep_rel']
        if dep_rel in relations:
            relations[dep_rel].append(token)
    return relations

def explain_ud_tags():
    explanations = {
        'nsubj': 'This is the nominal subject of the sentence, i.e., the noun or pronoun performing the action.',
        'root': 'This is the main verb (root) of the sentence, representing the core action.',
        'dobj': 'This is the direct object, the noun or pronoun that receives the action of the verb.'
    }
    return explanations

file_path = 'RuEval2017-Lenta-news-dev.conllu'

parsed_data = parse_conllu(file_path)

relations = analyze_syntactic_relations(parsed_data)

ud_tags_explanations = explain_ud_tags()

for dep_rel, tokens in relations.items():
    print(f"\nTokens with '{dep_rel}' relation:")
    print(ud_tags_explanations.get(dep_rel, "Explanation not available"))
    for token in tokens:
        print(f"Token ID: {token['token_id']}, Word: {token['word']}, Lemma: {token['lemma']}, " f"UPOS: {token['upos']}, Head: {token['head']}, Dependency Relation: {token['dep_rel']}")


def preview_conllu_file(filename, start_line=467, num_lines=30):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        preview = lines[start_line:start_line + num_lines]
        return ''.join(preview)

preview = preview_conllu_file('RuEval2017-Lenta-news-dev.conllu')
print(preview)

#2. Написать функцию разбиения сложносочиненного предложения из двух частей на простые.

def split_compound_sentence(sentence):
    doc = nlp(sentence)
    parts = []
    current_part = []

    for token in doc:
        if token.pos_ == "CCONJ" or token.text in [",", ";", ":"]:
            if current_part:
                parts.append(" ".join([t.text for t in current_part]))
                current_part = []
            else:
                current_part.append(token)

    if current_part:
        parts.append(" ".join([t.text for t in current_part]))

    return parts

test_sentence = "На нашей кухне нашли много всего: бактерии, грибки и позавчерашнее молоко."

result = split_compound_sentence(test_sentence)
print(f"original: {test_sentence}")
print(f"split: {result}")

#3. Написать функцию нахождения наименьшего общего предка двух токенов в дереве зависимостей

def find_lowest_common_ancestor(sentence, token1_idx, token2_idx):
    doc = nlp(sentence)
    token1 = doc[token1_idx] # indexes
    token2 = doc[token2_idx]
    path1 = []
    current = token1
    while current.head != current: # root
        path1.append(current)
        current = current.head
    path1.append(current)

    path2 = []
    current = token2
    while current.head != current:
        path2.append(current)
        current = current.head
    path2.append(current)

    for ancestor in path1:
        if ancestor in path2:
            return ancestor

test_sentence = "На нашей кухне нашли много всего: бактерии, грибки и позавчерашнее молоко."
token1_idx = 1
token2_idx = 5

lca = find_lowest_common_ancestor(test_sentence, token1_idx, token2_idx)
print(f"sentence: {test_sentence}")
print(f"tokens: '{test_sentence.split()[token1_idx]}' and '{test_sentence.split()[token2_idx]}'")
print(f"lowest ancestor: {lca.text} (POS: {lca.pos_}, DEP: {lca.dep_})")

#4. Сравнить три пары предложений двумя методами: сравнением расстояния редактирования деревьев зависимостей и косинусной мерой между BERT-эмбеддингами

# Загружаем модель для анализа зависимостей
nlp = spacy.load('ru_core_news_lg')

# Функция для получения дерева зависимостей из текста
def get_dependency_tree(text):
    doc = nlp(text)
    dependencies = []
    for token in doc:
        dependencies.append((token.dep_, token.head.text, token.text))
    return dependencies

# Функция для вычисления расстояния редактирования деревьев зависимостей
def edit_distance_trees(tree1, tree2):
    edits = 0
    for rel1, head1, dep1 in tree1:
        found_match = False
        for rel2, head2, dep2 in tree2:
            if dep1 == dep2 and rel1 == rel2 and head1 == head2:
                found_match = True
                break
        if not found_match:
            edits += 1
    return edits

# БERT модель для получения эмбеддингов
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Функция для получения эмбеддинга предложения
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Косинусная мера для сравнения BERT-эмбеддингов
def cosine_similarity_score(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

examples = [
"Привет, у нас на кухне нашли плесень!",
"На нашей кухне нашли много всего: бактерии, грибки и позавчерашнее молоко.",
"Привет, у них в подвале нашли клад!"
]

# Сравнение предложений с помощью деревьев зависимостей
print("Сравнение с помощью деревьев зависимостей:")
for i in range(len(examples)):
    for j in range(i + 1, len(examples)):
        sent1 = examples[i]
        sent2 = examples[j]

        tree1 = get_dependency_tree(sent1)
        tree2 = get_dependency_tree(sent2)
        edit_distance = edit_distance_trees(tree1, tree2)
        print(f"Редакционное расстояние между '{sent1}' и '{sent2}': {edit_distance}")

# Сравнение предложений с помощью косинусной меры BERT-эмбеддингов
print("\nСравнение с помощью косинусной меры BERT-эмбеддингов:")
for i in range(len(examples)):
    for j in range(i + 1, len(examples)):
        sent1 = examples[i]
        sent2 = examples[j]
        embedding1 = get_bert_embedding(sent1)
        embedding2 = get_bert_embedding(sent2)
        cosine_sim = cosine_similarity_score(embedding1, embedding2)
        print(f"Косинусное сходство между '{sent1}' и '{sent2}': {cosine_sim:.4f}")




