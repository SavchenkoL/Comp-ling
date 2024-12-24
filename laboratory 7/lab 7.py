import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Загрузка данных
data = pd.read_csv('data.csv')  # Замените на ваш файл с данными
texts = data['text']
labels = data['label']

# 1. Использование внешнего словаря тональностей
def load_sentiment_dictionary(file_path):
    sentiment_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            word, sentiment = line.strip().split()
            sentiment_dict[word] = sentiment
    return sentiment_dict

sentiment_dict = load_sentiment_dictionary('sentiment_dict.txt')  # Замените на ваш файл

# 2. Улучшение качества модели
def feature_extraction(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    sentiment_score = sum([1 if token in sentiment_dict else 0 for token in lemmatized_tokens])
    return ' '.join(lemmatized_tokens), sentiment_score

features = [feature_extraction(text) for text in texts]
X, sentiment_scores = zip(*features)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 3. Сравнение качества классификации
# Обучение fasttext-классификатора
with open('train.txt', 'w') as f:
    for text, label in zip(X_train, y_train):
        f.write(f'__label__{label} {text}\n')

model = fasttext.train_supervised('train.txt')

# 4. Оценка качества классификации
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred[0]))
