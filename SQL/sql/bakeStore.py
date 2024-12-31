"""
Создано: 31.12.2024

@Автор: Лейман М.А.
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, Date, DECIMAL, func


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from bikeStoreQueries import BikeStoreQueries


def main():
    try:
        # Подключение к базе данных через SQLAlchemy
        engine = create_engine('postgresql://postgres:postgres@localhost/BikeStore')
        Session = sessionmaker(bind=engine)
        session = Session()

        bike_store_queries = BikeStoreQueries(session)

        while True:
            # Меню выбора действий
            print("\nВыберите действие:")
            print("1: Продукты и торговые марки")
            print("2: Активные сотрудники и их магазины")
            print("3: Покупатели магазина")
            print("4: Количество продуктов в каждой категории")
            print("5: Количество заказов для каждого клиента")
            print("6: Клиенты, которые сделали хотя бы 1 заказ")
            print("0: Выход")

            try:
                choice = int(input("\nВведите номер действия: "))
            except ValueError:
                print("\nПожалуйста, введите корректный номер.")
                continue

            if choice == 0:
                print("\nВыход из программы.")
                break

            if choice == 1:
                # 1. Продукты и торговые марки
                display_results("\n1. Продукты и торговые марки", bike_store_queries.get_products_and_brands())
            elif choice == 2:
                # 2. Активные сотрудники и их магазины
                display_results("2. Активные сотрудники и их магазины", bike_store_queries.get_active_employees_and_stores())
            elif choice == 3:
                # 3. Покупатели магазина
                try:
                    store_id = int(input("Введите ID магазина: "))
                    store_name, customers_by_store = bike_store_queries.get_customers_by_store(store_id)
                    if store_name:
                        print(f"\n3. Покупатели магазина '{store_name}' (ID: {store_id}):\n")
                        display_results(None, customers_by_store, enumerate_results=True)
                    else:
                        print(f"\nМагазин с ID {store_id} не найден.")
                except ValueError:
                    print("Пожалуйста, введите корректный ID магазина.")
            elif choice == 4:
                # 4. Количество продуктов в каждой категории
                display_results("4. Количество наименований продуктов в каждой категории", bike_store_queries.get_product_count_by_category())
            elif choice == 5:
                # 5. Количество заказов для каждого клиента
                display_results("5. Количество заказов для каждого клиента", bike_store_queries.get_order_count_per_customer())
            elif choice == 6:
                # 6. Клиенты, которые сделали хотя бы 1 заказ
                display_results("6. Клиенты, которые сделали хотя бы 1 заказ", bike_store_queries.get_customers_with_orders())
            else:
                print("Некорректный выбор. Попробуйте снова.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        session.close()


def display_results(title, results, enumerate_results=False):
    """
    Утилита для вывода результатов.
    :param title: Заголовок секции
    :param results: Список данных для вывода
    :param enumerate_results: Если True, добавляет порядковый номер к результатам
    """
    if title:
        print(f"\n{title}:\n")
    if results:
        i = 0
        for row in results:
            i += 1
            print(i, row)
    else:
        print("Нет данных для отображения.")


if __name__ == '__main__':
    main()


