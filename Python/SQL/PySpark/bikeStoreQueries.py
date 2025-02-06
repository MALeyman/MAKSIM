
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.sql.functions import year, month, sum
from pyspark.sql import functions as F

class BikeStoreQueries:
    def __init__(self, spark, db_url, db_properties):
        """
        Инициализирует объект BikeStoreQueries для работы с PySpark и PostgreSQL.

        Args:
            spark (SparkSession): Экземпляр SparkSession.
            db_url (str): JDBC URL для подключения к базе данных PostgreSQL.
            db_properties (dict): Свойства базы данных (пользователь, пароль и драйвер).
        """
        self.spark = spark
        self.db_url = db_url
        self.db_properties = db_properties

    def load_table(self, table_name):
        """
        Загружает таблицу из базы данных в DataFrame.
        """
        return self.spark.read.jdbc(self.db_url, table_name, properties=self.db_properties)

    def get_products_and_brands(self):
        """
        Возвращает список всех продуктов с их торговыми марками.
        """
        products = self.load_table("products")
        brands = self.load_table("brands")
        query = products.join(brands, products.brand_id == brands.brand_id) \
                        .select(products.product_name, brands.brand_name)
        return query

    def get_active_employees_and_stores(self):
        """
        Возвращает список всех активных сотрудников и названия магазинов, в которых они работают.
        """
        staffs = self.load_table("staffs")
        stores = self.load_table("stores")
        query = staffs.filter(col("active") == 1) \
                      .join(stores, staffs.store_id == stores.store_id) \
                      .select(staffs.first_name, staffs.last_name, stores.store_name)
        return query

    def get_customers_by_store(self, store_id):
        """
        Возвращает название магазина и список всех покупателей, связанных с указанным ID магазина.
        """
        stores = self.load_table("stores")
        customers = self.load_table("customers")
        orders = self.load_table("orders")

        # Получить название магазина
        store_name_df = stores.filter(col("store_id") == store_id).select("store_name").collect()
        if not store_name_df:
            return None, None

        store_name = store_name_df[0]["store_name"]

        # Получить покупателей
        query = customers.join(orders, customers.customer_id == orders.customer_id) \
                        .filter(col("store_id") == store_id) \
                        .select(customers.first_name, customers.last_name, customers.email, customers.phone)
        return store_name, query

    def get_product_count_by_category(self):
        """
        Возвращает количество продуктов в каждой категории.
        """
        categories = self.load_table("categories")
        products = self.load_table("products")
        query = products.join(categories, products.category_id == categories.category_id) \
                        .groupBy(categories.category_name) \
                        .agg(count(products.product_id).alias("product_count"))
        return query

    def get_order_count_per_customer(self):
        """
        Возвращает количество заказов для каждого клиента.
        """
        customers = self.load_table("customers")
        orders = self.load_table("orders")
        query = customers.join(orders, customers.customer_id == orders.customer_id) \
                         .groupBy(customers.first_name, customers.last_name) \
                         .agg(count(orders.order_id).alias("order_count"))
        return query

    def get_customers_with_orders(self):
        """
        Возвращает список клиентов, которые сделали хотя бы один заказ, и общее количество их заказов.
        """
        customers = self.load_table("customers")
        orders = self.load_table("orders")
        query = customers.join(orders, customers.customer_id == orders.customer_id) \
                         .groupBy(customers.first_name, customers.last_name) \
                         .agg(count(orders.order_id).alias("order_count")) \
                         .filter(col("order_count") > 0)
        return query

    def get_total_sales_per_product(self):
        """
        Расчет общего объема продаж по каждому продукту.
        """
        products = self.load_table("products")
        order_items = self.load_table("order_items")  # Мы предполагаем, что у нас есть таблица order_items
        query = order_items.join(products, order_items.product_id == products.product_id) \
                        .select(order_items.quantity, order_items.list_price, products.product_name, order_items.discount) \
                        .withColumn("total_sales", (order_items.quantity * order_items.list_price * (1 - order_items.discount)))
        total_sales_query = query.groupBy("product_name").agg(sum("total_sales").alias("total_sales"))
        return total_sales_query

    def get_order_count_by_status(self):
        """
        Расчет количества заказов по каждому статусу заказа.
        """
        orders = self.load_table("orders")
        query = orders.groupBy("order_status").agg(count(orders.order_id).alias("order_count"))
        return query


    def get_total_sales_per_month(self):
        """
        Расчет общей суммы продаж за каждый месяц.
        """
        orders = self.load_table("orders")
        order_items = self.load_table("order_items")
        products = self.load_table("products")
        
        # Присоединим таблицы, чтобы получить цену и количество для каждого заказа
        query = order_items.join(orders, order_items.order_id == orders.order_id) \
                            .join(products, order_items.product_id == products.product_id) \
                            .select(orders.order_date, order_items.quantity, order_items.list_price, order_items.discount) \
                            .withColumn("total_sales", (order_items.quantity * order_items.list_price * (1 - order_items.discount))) \
                            .withColumn("year", F.year("order_date")) \
                            .withColumn("month", F.month("order_date"))
        
        # Группируем по месяцу и году
        total_sales_query = query.groupBy("year", "month").agg(F.sum("total_sales").alias("total_sales"))
        
        # Сортируем по году, а затем по месяцу
        sorted_sales_query = total_sales_query.orderBy("year", "month")
        
        return sorted_sales_query


    from pyspark.sql.functions import sum

    def get_top_5_customers_by_spending(self):
        """
        Топ 5 клиентов, которые потратили больше всего денег.
        """
        orders = self.load_table("orders")
        order_items = self.load_table("order_items")
        customers = self.load_table("customers")
        products = self.load_table("products")
        
        # Присоединим таблицы, чтобы получить цены и количества товаров
        query = order_items.join(orders, order_items.order_id == orders.order_id) \
                        .join(products, order_items.product_id == products.product_id) \
                        .join(customers, orders.customer_id == customers.customer_id) \
                        .select(customers.first_name, customers.last_name, order_items.quantity, order_items.list_price, order_items.discount) \
                        .withColumn("total_spent", (order_items.quantity * order_items.list_price * (1 - order_items.discount)))
        
        # Суммируем расходы по каждому клиенту и сортируем по убыванию
        total_spent_query = query.groupBy("first_name", "last_name").agg(sum("total_spent").alias("total_spent")) \
                                .orderBy(col("total_spent").desc())
        
        return total_spent_query.limit(5)

    
    
