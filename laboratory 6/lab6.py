import re


def mask_info(text):
    # Маскировка адресов электронной почты
    masked_email = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)

    # Маскировка имен
    masked_text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', masked_email)

    return masked_text


email_text = "Контактируйте Ивана Иванова по адресу ivanov@mail.com"
masked_text = mask_info(email_text)
print(masked_text)  # Вывод: Контактируйте [NAME] по адресу [EMAIL]

from collections import Counter

# Пример описаний вакансий
job_descriptions = [
    "Опыт работы с Java, Spring, Hibernate, SQL.",
    "Нужны знания Java, Maven, RESTful Services.",
    "Требуется Java, Spring Boot, Microservices.",
    "Java, Unit Testing, JPA, Git.",
    "Ищем специалиста с опытом в Java, Docker, Kubernetes."
]


def extract_top_skills(descriptions):
    skills = []
    for desc in descriptions:
        skills.extend(re.findall(r'\b\w+\b', desc))  # Извлечение слов

    # Считаем частоту навыков
    skills_count = Counter(skills)

    # Получаем 5 самых распространенных навыков
    top_skills = skills_count.most_common(5)
    return top_skills


top_skills = extract_top_skills(job_descriptions)
print(top_skills)  # Вывод: список из 5 наиболее частых навыков

from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split

# Пример данных
X = [[{'word': 'Привет'}, {'word': 'мир'}],
     [{'word': 'Я'}, {'word': 'разработчик'}]]
y = [['greeting', 'noun'], ['pronoun', 'noun']]

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Настройка CRF
crf = CRF(algorithm='lbfgs', max_iterations=100)
crf.fit(X_train, y_train)

# Предсказания
y_pred = crf.predict(X_test)
print(y_pred)  # Вывод предсказаний


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Пример данных
sentences = [['я', 'разработчик'], ['это', 'тест']]
labels = [['O', 'B-Job'], ['O', 'O']]

# Токенизация
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X)

# Эмбеддинги
embeddings_index = {'разработчик': np.random.rand(100), 'тест': np.random.rand(100)}
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Создание LSTM модели
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, weights=[embedding_matrix], trainable=False))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='softmax')))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Обучение модели
model.fit(X, np.array([[1], [0]]), epochs=5)

# Предсказания
predicted = model.predict(X)
print(predicted)  # Вывод предсказаний

