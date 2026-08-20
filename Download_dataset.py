import os
import io
import time
import requests

from PIL import Image
from ddgs import DDGS

from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# НАСТРОЙКИ
# =========================================================

DATASET_PATH = "Dataset"

# Сколько изображений должно быть в каждом классе
IMAGES_PER_CLASS = 300

# Сколько загрузок одновременно
MAX_WORKERS = 10

# Минимальный размер изображения
MIN_WIDTH = 150
MIN_HEIGHT = 150

# Сколько результатов получать из каждого поиска
SEARCH_RESULTS = 500

# =========================================================
# ПОИСКОВЫЕ ЗАПРОСЫ
# =========================================================

SEARCHES = {

    "cats": [
        "cat animal photo",
        "cat portrait photo",
        "cat sitting photo",
        "cat standing photo",
        "cat full body photo",
        "cat outdoor photo",
        "cat indoor photo",
        "kitten photo",
        "cute cat photo",
        "domestic cat photo",
    ],

    "dogs": [
        "dog animal photo",
        "dog portrait photo",
        "dog sitting photo",
        "dog standing photo",
        "dog full body photo",
        "dog outdoor photo",
        "dog indoor photo",
        "puppy photo",
        "small dog photo",
        "pomeranian dog photo",
        "spitz dog photo",
        "fluffy dog photo",
    ],

    "rats": [
        "rat animal photo",
        "rat portrait photo",
        "rat sitting photo",
        "rat standing photo",
        "rat full body photo",
        "pet rat photo",
        "brown rat photo",
        "white rat photo",
        "fancy rat photo",
        "domestic rat photo",
    ]
}


# =========================================================
# СОЗДАНИЕ ПАПОК
# =========================================================

os.makedirs(DATASET_PATH, exist_ok=True)

for class_name in SEARCHES:

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    os.makedirs(
        folder,
        exist_ok=True
    )


# =========================================================
# ПОЛУЧЕНИЕ КОЛИЧЕСТВА ФАЙЛОВ
# =========================================================

def get_existing_images(folder):

    files = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            files.append(filename)

    return files


# =========================================================
# ПОЛУЧЕНИЕ СЛЕДУЮЩЕГО НОМЕРА
# =========================================================

def get_next_number(folder, class_name):

    numbers = []

    for filename in os.listdir(folder):

        if not filename.lower().endswith(".jpg"):
            continue

        if not filename.startswith(
            class_name + "_"
        ):
            continue

        try:

            number = int(
                os.path.splitext(filename)[0]
                .split("_")[-1]
            )

            numbers.append(number)

        except ValueError:

            continue

    if not numbers:
        return 1

    return max(numbers) + 1


# =========================================================
# СКАЧИВАНИЕ ОДНОГО ИЗОБРАЖЕНИЯ
# =========================================================

def download_image(task):

    url, filepath = task

    try:

        headers = {
            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/151.0 Safari/537.36"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return False

        # Проверяем, что это изображение
        image_data = io.BytesIO(
            response.content
        )

        image = Image.open(
            image_data
        )

        # Проверяем изображение
        image.verify()

        # Открываем заново
        image_data.seek(0)

        image = Image.open(
            image_data
        )

        width, height = image.size

        # Отбрасываем слишком маленькие изображения
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False

        # RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Сохраняем
        image.save(
            filepath,
            "JPEG",
            quality=92
        )

        return True

    except Exception:

        return False


# =========================================================
# ПОЛУЧЕНИЕ URL ИЗ ПОИСКА
# =========================================================

def get_search_urls(search_query):

    urls = []

    print(
        f"Поиск: {search_query}"
    )

    try:

        with DDGS() as ddgs:

            results = ddgs.images(
                search_query,
                max_results=SEARCH_RESULTS
            )

            for result in results:

                url = result.get("image")

                if url:
                    urls.append(url)

    except Exception as e:

        print(
            f"Ошибка поиска: {e}"
        )

    return urls


# =========================================================
# СКАЧИВАНИЕ КЛАССА
# =========================================================

def download_class(
    class_name,
    searches
):

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    existing = get_existing_images(
        folder
    )

    current_count = len(existing)

    print()
    print("=" * 60)
    print(
        f"КЛАСС: {class_name}"
    )
    print(
        f"Уже скачано: {current_count}"
    )
    print(
        f"Цель: {IMAGES_PER_CLASS}"
    )
    print("=" * 60)

    # -----------------------------------------------------
    # Уже достаточно
    # -----------------------------------------------------

    if current_count >= IMAGES_PER_CLASS:

        print(
            "Цель уже достигнута!"
        )

        return

    # -----------------------------------------------------
    # Следующий номер
    # -----------------------------------------------------

    next_number = get_next_number(
        folder,
        class_name
    )

    # -----------------------------------------------------
    # Сначала собираем URL
    # -----------------------------------------------------

    all_urls = []

    for search_query in searches:

        if len(all_urls) >= (
            IMAGES_PER_CLASS * 3
        ):
            break

        urls = get_search_urls(
            search_query
        )

        all_urls.extend(urls)

        print(
            f"Получено URL: {len(urls)}"
        )

    # Убираем дубликаты
    all_urls = list(
        dict.fromkeys(all_urls)
    )

    print()
    print(
        f"Всего уникальных URL: "
        f"{len(all_urls)}"
    )

    # -----------------------------------------------------
    # Сколько нужно скачать
    # -----------------------------------------------------

    needed = (
        IMAGES_PER_CLASS -
        current_count
    )

    print(
        f"Нужно ещё: {needed}"
    )

    # -----------------------------------------------------
    # Создаём задачи
    # -----------------------------------------------------

    tasks = []

    for url in all_urls:

        if len(tasks) >= (
            needed * 3
        ):
            break

        filepath = os.path.join(
            folder,
            f"{class_name}_{next_number + len(tasks):04d}.jpg"
        )

        tasks.append(
            (url, filepath)
        )

    # -----------------------------------------------------
    # Параллельная загрузка
    # -----------------------------------------------------

    successful = 0

    print()
    print(
        f"Начинаем загрузку "
        f"({MAX_WORKERS} потоков)..."
    )
    print()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_task = {
            executor.submit(
                download_image,
                task
            ): task

            for task in tasks
        }

        for future in as_completed(
            future_to_task
        ):

            url, filepath = (
                future_to_task[future]
            )

            try:

                success = future.result()

            except Exception:

                success = False

            if success:

                successful += 1

                print(
                    f"[{successful}/{needed}] "
                    f"OK -> "
                    f"{os.path.basename(filepath)}"
                )

            # -------------------------------------------------
            # Достигли нужного количества
            # -------------------------------------------------

            if successful >= needed:

                break

    # -----------------------------------------------------
    # Итог
    # -----------------------------------------------------

    final_count = len(
        get_existing_images(folder)
    )

    print()
    print(
        f"{class_name}: "
        f"{final_count} изображений"
    )


# =========================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =========================================================

def main():

    print()
    print("=" * 60)
    print("       FAST DATASET DOWNLOADER")
    print("=" * 60)
    print()

    print(
        f"Цель: {IMAGES_PER_CLASS} "
        f"изображений каждого класса"
    )

    print(
        f"Одновременных загрузок: "
        f"{MAX_WORKERS}"
    )

    print()

    # =====================================================
    # ВАЖНО:
    # Существующие изображения НЕ удаляются.
    # =====================================================

    for class_name, searches in (
        SEARCHES.items()
    ):

        download_class(
            class_name,
            searches
        )

    # =====================================================
    # ФИНАЛЬНАЯ СТАТИСТИКА
    # =====================================================

    print()
    print("=" * 60)
    print("ГОТОВО!")
    print("=" * 60)

    print()

    total = 0

    for class_name in SEARCHES:

        folder = os.path.join(
            DATASET_PATH,
            class_name
        )

        count = len(
            get_existing_images(folder)
        )

        total += count

        print(
            f"{class_name}: {count}"
        )

    print()
    print(
        f"Всего изображений: {total}"
    )

    print()
    print(
        "Dataset готов!"
    )


# =========================================================
# ЗАПУСК
# =========================================================

if __name__ == "__main__":
    main()