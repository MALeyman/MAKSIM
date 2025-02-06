from sparkSession import get_spark_session
# from bikeStoreQueries import BikeStoreQueries

from bikeStoreQueries import BikeStoreQueries

def main():
    spark = get_spark_session()
    jdbc_url = "jdbc:postgresql://localhost:5432/BikeStore"
    db_properties = {
        "user": "postgres",
        "password": "postgres",
        "driver": "org.postgresql.Driver"
    }

    bike_store_queries = BikeStoreQueries(spark, jdbc_url, db_properties)

    while True:
        print("\nВыберите действие:")
        print("1: Продукты и торговые марки")
        print("2: Активные сотрудники и их магазины")
        print("3: Покупатели магазина")
        print("4: Количество продуктов в каждой категории")
        print("5: Количество заказов для каждого клиента")
        print("6: Клиенты, которые сделали хотя бы 1 заказ")
        print("7: Общий объем продаж по каждому продукту")
        print("8: Количество заказов по каждому статусу заказа")
        print("9: Общая сумма продаж за каждый месяц")
        print("10: Топ 5 клиентов, которые потратили больше всего денег")
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
            display_results("1. Продукты и торговые марки", bike_store_queries.get_products_and_brands())
        elif choice == 2:
            display_results("2. Активные сотрудники и их магазины", bike_store_queries.get_active_employees_and_stores())
        elif choice == 3:
            try:
                store_id = int(input("Введите ID магазина: "))
                store_name, customers_by_store = bike_store_queries.get_customers_by_store(store_id)
                if store_name:
                    print(f"\n3. Покупатели магазина '{store_name}' (ID: {store_id}):\n")
                    display_results(None, customers_by_store)
                else:
                    print(f"\nМагазин с ID {store_id} не найден.")
            except ValueError:
                print("Пожалуйста, введите корректный ID магазина.")
        elif choice == 4:
            display_results("4. Количество наименований продуктов в каждой категории", bike_store_queries.get_product_count_by_category())
        elif choice == 5:
            display_results("5. Количество заказов для каждого клиента", bike_store_queries.get_order_count_per_customer())
        elif choice == 6:
            display_results("6. Клиенты, которые сделали хотя бы 1 заказ", bike_store_queries.get_customers_with_orders())
        elif choice == 7:
            display_results("7. Общий объем продаж по каждому продукту", bike_store_queries.get_total_sales_per_product())
        elif choice == 8:
            display_results("8. Количество заказов по каждому статусу заказа", bike_store_queries.get_order_count_by_status())
        elif choice == 9:
            display_results("9. Общая сумма продаж за каждый месяц", bike_store_queries.get_total_sales_per_month())
        elif choice == 10:
            display_results("10. Топ 5 клиентов, которые потратили больше всего денег", bike_store_queries.get_top_5_customers_by_spending())
        else:
            print("Некорректный выбор. Попробуйте снова.")

    
    
    
                
def display_results(title, results):
    """
    Утилита для вывода результатов.
    :param title: Заголовок секции.
    :param results: DataFrame с результатами.
    :param enumerate_results: Если True, добавляет порядковый номер к результатам.
    """
    if title:
        print(f"\n{title}:\n")
    if results is not None:
        results.show(n=results.count(), truncate=False)
        # results.show(truncate=False)
    else:
        print("Нет данных для отображения.")         
                
                

if __name__ == "__main__":
    main()
