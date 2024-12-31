
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

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
