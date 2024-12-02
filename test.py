import requests
import pandas as pd


# Функция для анализа вакансии
def analyze_vacancy(item):
    analysis = f"Вакансия {item['name']} в компании {item['employer']['name']} "
    if item.get('salary'):
        salary_from = item['salary']['from'] if item['salary']['from'] else 'не указана'
        analysis += f"с зарплатой от {salary_from} рублей. "
    analysis += "Работа требует опыта от 1 до 3 лет."
    return analysis


# Функция для получения вакансий по заданному тексту поиска
def get_vacancies(search_text, area):
    url = 'https://api.hh.ru/vacancies'
    params = {
        'area': area,
        'text': search_text,
        'experience': 'between1And3',
        'employment': 'full',
    }
    response = requests.get(url, params=params)
    return response.json()


# Список всех категорий с вакансиями
job_categories = [
    'Программист',
    'Системный администратор',
    'Аналитик данных',
    'Специалист по машинному обучению',
    'Специалист по контент-маркетингу',
    'SMM-менеджер',
    'SEO-специалист',
    'Финансовый аналитик',
    'Бухгалтер',
    'Кредитный аналитик',
    'Оператор колл-центра',
    'Специалист по поддержке пользователей',
    'Менеджер по продажам'
]

# Список регионов (пример для других регионов России)
regions = [1, 2, 3, 4]  # Номера регионов (например, 1 - Москва, 2 - Санкт-Петербург и т.д.)

# Извлечение вакансий для всех категорий
vacancies = []
for area in regions:
    for category in job_categories:
        data = get_vacancies(category, area)
        for item in data['items']:
            # Упрощаем определение требований
            requirements = item['snippet']['requirement'] if 'snippet' in item and 'requirement' in item[
                'snippet'] else None
            # Упрощаем определение задач
            tasks = item['snippet']['responsibility'] if 'snippet' in item and 'responsibility' in item[
                'snippet'] else None

            # Обработка требований и задач
            simplified_requirements = requirements if requirements else 'не указаны'
            simplified_tasks = tasks if tasks else 'не указаны'

            # Сбор данных о вакансии
            vacancy_data = {
                'title': item['name'],  # Заголовок вакансии
                'company': item['employer']['name'],  # Название компании
                'link': item['alternate_url'],  # Ссылка на вакансию
                'city': item['area']['name'],  # Город расположения вакансии
                'salary': item['salary']['from'] if item['salary'] else None,  # Зарплата (если указана)
                'experience': '1-3 года',  # Требуемый опыт работы
                'requirements': simplified_requirements,  # Полные требования к вакансии
                'tasks': simplified_tasks,  # Задачи от работодателя
                'analysis': analyze_vacancy(item)  # Краткий анализ вакансии
            }
            vacancies.append(vacancy_data)

# Сохранение данных в Excel
df = pd.DataFrame(vacancies)
df.to_excel('vacancies.xlsx', index=False)  # Сохранение в формате Excel
