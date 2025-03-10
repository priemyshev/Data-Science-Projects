import flet as ft
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Загрузка датасета с указанием конфигурации 'main'
dataset = load_dataset("openai/gsm8k", 'main', split="train")

# Инициализация модели embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Создание Qdrant клиента в памяти
client = QdrantClient(":memory:")

# Создание отдельных коллекций для вопросов и ответов
questions_collection = "math_questions"
answers_collection = "math_answers"

# Создание коллекций
for collection_name in [questions_collection, answers_collection]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Подготовка и загрузка векторов
def prepare_vectors():
    question_points = []
    answer_points = []
    
    for idx, item in enumerate(dataset):
        # Создаем отдельные векторы для вопросов и ответов
        question_vector = model.encode(item['question'])
        answer_vector = model.encode(item['answer'])
        
        question_points.append(PointStruct(
            id=idx, 
            vector=question_vector.tolist(), 
            payload={
                'question': item['question'], 
                'answer': item['answer']
            }
        ))
        
        answer_points.append(PointStruct(
            id=idx, 
            vector=answer_vector.tolist(), 
            payload={
                'question': item['question'], 
                'answer': item['answer']
            }
        ))
    
    # Загрузка векторов в соответствующие коллекции
    client.upsert(collection_name=questions_collection, points=question_points)
    client.upsert(collection_name=answers_collection, points=answer_points)

# Подготовка векторов
prepare_vectors()

def main(page: ft.Page):
    page.title = "Поиск математических задач GSM8K"
    page.window_width = 800
    page.window_height = 900
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20

    # Поля ввода
    question_input = ft.TextField(
        label="Поиск по вопросу", 
        width=700,
        hint_text="Введите текст для поиска по вопросам...",
        multiline=True,
        min_lines=1,
        max_lines=3
    )
    answer_input = ft.TextField(
        label="Поиск по ответу", 
        width=700,
        hint_text="Введите текст для поиска по ответам...",
        multiline=True,
        min_lines=1,
        max_lines=3
    )

    # Создаем прокручиваемый контейнер для результатов
    results_container = ft.Container(
        content=ft.Column([], scroll=ft.ScrollMode.AUTO),
        height=500,
        width=750
    )

    def search_by_field(e, query_text, collection_name):
        # Очистка предыдущих результатов
        results_container.content.controls.clear()

        if not query_text.strip():
            results_container.content.controls.append(
                ft.Text("Пожалуйста, введите текст для поиска", size=16)
            )
            page.update()
            return

        # Создание вектора запроса
        query_vector = model.encode(query_text).tolist()

        # Поиск в Qdrant
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10
        )

# Отображение результатов
        if not search_result:
            results_container.content.controls.append(
                ft.Text("Результаты не найдены", size=16)
            )
        else:
            for result in search_result:
                result_card = ft.Container(
                    content=ft.Column([
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Вопрос:", size=16, weight=ft.FontWeight.BOLD),
                                ft.Text(result.payload['question'], 
                                       size=14, 
                                       selectable=True,
                                       text_align=ft.TextAlign.JUSTIFY),
                                ft.Text("Ответ:", size=16, weight=ft.FontWeight.BOLD),
                                ft.Text(result.payload['answer'], 
                                       size=14, 
                                       selectable=True,
                                       text_align=ft.TextAlign.JUSTIFY),
                                ft.Text(f"Схожесть: {result.score:.2f}", 
                                       size=12, 
                                       color=ft.colors.GREY_700)
                            ]),
                            padding=15
                        ),
                        ft.Divider(height=1)
                    ]),
                    bgcolor=ft.colors.BLUE_50,
                    border_radius=10,
                    margin=ft.margin.only(bottom=10)
                )
                results_container.content.controls.append(result_card)
        
        page.update()

    def search_questions(e):
        search_by_field(e, question_input.value, questions_collection)

    def search_answers(e):
        search_by_field(e, answer_input.value, answers_collection)

    # Кнопки поиска
    question_search_button = ft.ElevatedButton(
        "Искать по вопросам",
        on_click=search_questions,
        style=ft.ButtonStyle(
            bgcolor=ft.colors.BLUE,
            color=ft.colors.WHITE,
        )
    )

    answer_search_button = ft.ElevatedButton(
        "Искать по ответам",
        on_click=search_answers,
        style=ft.ButtonStyle(
            bgcolor=ft.colors.GREEN,
            color=ft.colors.WHITE,
        )
    )

    # Компоновка интерфейса
    page.add(
        ft.Column([
            ft.Text("Поиск математических задач GSM8K", 
                   size=30, 
                   weight=ft.FontWeight.BOLD,
                   text_align=ft.TextAlign.CENTER),
            ft.Container(
                content=ft.Column([
                    question_input,
                    question_search_button,
                ]),
                padding=10
            ),
            ft.Container(
                content=ft.Column([
                    answer_input,
                    answer_search_button,
                ]),
                padding=10
            ),
            ft.Container(
                content=results_container,
                padding=10,
                border=ft.border.all(1, ft.colors.GREY_300),
                border_radius=10
            )
        ], 
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20)
    )

# Запуск приложения
ft.app(target=main)