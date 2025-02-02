

import pytest

class BooksCollector:
    def __init__(self):
        self.books_genre = {}
        self.favorites = []
        self.genre = ["Фантастика", "Фэнтези", "Научная фантастика", "Детектив", "Роман"]
        self.genre_age_rating = {"Фантастика": 12, "Фэнтези": 10, "Научная фантастика": 14, "Детектив": 16, "Роман": 18}

    def add_new_book(self, book_name):
        if book_name not in self.books_genre:
            self.books_genre[book_name] = None
    
    def set_book_genre(self, book_name, genre):
        if book_name in self.books_genre and genre in self.genre:
            self.books_genre[book_name] = genre

    def get_book_genre(self, book_name):
        return self.books_genre.get(book_name)

    def get_books_with_specific_genre(self, genre):
        return [book for book, book_genre in self.books_genre.items() if book_genre == genre]

    def get_books_genre(self):
        return self.books_genre

    def get_books_for_children(self):
        return [book for book, genre in self.books_genre.items() if genre and self.genre_age_rating.get(genre, 18) <= 12]

    def add_book_in_favorites(self, book_name):
        if book_name in self.books_genre and book_name not in self.favorites:
            self.favorites.append(book_name)

    def delete_book_from_favorites(self, book_name):
        if book_name in self.favorites:
            self.favorites.remove(book_name)

    def get_list_of_favorites_books(self):
        return self.favorites

# Тесты
@pytest.fixture
def collector():
    return BooksCollector()

# Тест добавления книги
@pytest.mark.parametrize("book_name", ["Властелин колец", "Гарри Поттер"])
def test_add_new_book(collector: BooksCollector, book_name):
    '''
        Тест добавления книги
    '''
    print("Положительный тест: Тест добавления книги")
    collector.add_new_book(book_name)
    assert book_name in collector.books_genre
    assert collector.books_genre[book_name] is None

def test_add_existing_book(collector):
    '''
        Повторное добавление книги
    '''
    collector.add_new_book("Властелин колец")
    collector.add_new_book("Властелин колец")
    assert len(collector.books_genre) == 1
    print("Отрицательный тест: повторное добавление книги")

    
# Тест установки жанра
@pytest.mark.parametrize("book_name, genre", [("Гарри Поттер", "Фэнтези"), ("Матрица", "Научная фантастика")])
def test_set_book_genre(collector, book_name, genre):
    '''
        Тест установки жанра
    '''
    collector.add_new_book(book_name)
    collector.set_book_genre(book_name, genre)
    assert collector.get_book_genre(book_name) == genre
    print('Положительный тест: установки жанра')

def test_set_invalid_book_genre(collector):
    '''
        Установка жанра для несуществующей книги
    '''
    collector.set_book_genre("Несуществующая книга", "Фэнтези")
    assert collector.get_book_genre("Несуществующая книга") is None
    print("Отрицательный тест: установка жанра для несуществующей книги")

# Тест установки несуществующего жанра
def test_set_nonexistent_genre(collector):
    '''
        установка несуществующего жанра
    '''
    collector.add_new_book("Гарри Поттер")
    collector.set_book_genre("Гарри Поттер", "Мистика")
    assert collector.get_book_genre("Гарри Поттер") is None
    print("Отрицательный тест: установка несуществующего жанра")  

# Тест книг для детей
def test_get_books_for_children_empty(collector):
    '''
        Тест книг для детей: список детских книг пуст
    '''
    assert collector.get_books_for_children() == []
    print("Отрицательный тест: список детских книг пуст")

def test_get_books_for_children_with_books(collector):
    '''
        Тест книг для детей
    '''
    collector.add_new_book("Алиса в Стране чудес")
    collector.set_book_genre("Алиса в Стране чудес", "Фэнтези")
    assert collector.get_books_for_children() == ["Алиса в Стране чудес"]
    print("Положительный тест: книги для детей присутствуют")



def test_get_books_with_specific_genre(collector):
    '''
        Положительный тест: Получение книг с конкретным жанром
    '''
    collector.add_new_book("Гарри Поттер")
    collector.set_book_genre("Гарри Поттер", "Фэнтези")
    collector.add_new_book("1984")
    collector.set_book_genre("1984", "Детектив")
    
    # Проверяем, что метод вернул книги с жанром "Фэнтези"
    assert collector.get_books_with_specific_genre("Фэнтези") == ["Гарри Поттер"]
    print("Положительный тест: Получение книг с жанром 'Фэнтези'")

def test_get_books_with_nonexistent_genre(collector):
    '''
        Отрицательный тест: Получение книг с несуществующим жанром
    '''
    collector.add_new_book("Гарри Поттер")
    collector.set_book_genre("Гарри Поттер", "Фэнтези")
    
    # Проверяем, что метод вернул пустой список, так как жанр "Научная фантастика" не установлен
    assert collector.get_books_with_specific_genre("Научная фантастика") == []
    print("Отрицательный тест: Получение книг с несуществующим жанром")


def test_get_books_genre(collector):
    '''
        Положительный тест: Получение всех жанров книг
    '''
    collector.add_new_book("Гарри Поттер")
    collector.set_book_genre("Гарри Поттер", "Фэнтези")
    collector.add_new_book("1984")
    collector.set_book_genre("1984", "Детектив")
    
    # Проверяем, что метод возвращает правильный словарь с жанрами
    expected_genres = {
        "Гарри Поттер": "Фэнтези",
        "1984": "Детектив"
    }
    assert collector.get_books_genre() == expected_genres
    print("Положительный тест: Получение всех жанров книг")

def test_get_books_genre_empty(collector):
    '''
        Отрицательный тест: Получение жанров книг, когда книги не добавлены
    '''
    # Проверяем, что метод возвращает пустой словарь, так как книг нет
    assert collector.get_books_genre() == {}
    print("Отрицательный тест: Получение жанров книг, когда книги не добавлены")



# Тест добавления в избранное
@pytest.mark.parametrize("book_name", ["1984", "Гарри Поттер"])
def test_add_book_in_favorites(collector, book_name):
    '''
        Тест добавления в избранное
    '''
    print('Положительный тест: Тест добавления в избранное')
    collector.add_new_book(book_name)
    collector.add_book_in_favorites(book_name)
    assert book_name in collector.get_list_of_favorites_books()

def test_add_non_existing_book_in_favorites(collector):
    '''
        Добавление несуществующей книги в избранное
    '''
    collector.add_book_in_favorites("Несуществующая книга")
    assert "Несуществующая книга" not in collector.get_list_of_favorites_books()
    print("Отрицательный тест: добавление несуществующей книги в избранное")

# Тест удаления из избранного
@pytest.mark.parametrize("book_name", ["1984", "Гарри Поттер"])
def test_delete_book_from_favorites_existing(collector, book_name):
    '''
        Тест удаления из избранного
    '''
    collector.add_new_book(book_name)
    collector.add_book_in_favorites(book_name)
    collector.delete_book_from_favorites(book_name)
    assert book_name not in collector.get_list_of_favorites_books()
    print("Положительный тест: успешное удаление книги из избранного")

def test_delete_book_from_favorites_not_existing(collector):
    '''
        Тест удаления из избранного несуществующей книги
    '''
    collector.delete_book_from_favorites("Несуществующая книга")
    assert "Несуществующая книга" not in collector.get_list_of_favorites_books()
    print("Отрицательный тест: удаление книги, которой нет в избранном")

# Запуск покрытия тестами
if __name__ == "__main__":
    import pytest
    pytest.main()