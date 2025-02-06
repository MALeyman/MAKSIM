
"""
Создано: 31.12.2024

@Автор: Лейман М.А.
"""

from modelsTable import Store, Brand, Category, Product, Customer, Order, OrderItem, Stock, Staff
from sqlalchemy.orm import Session
from sqlalchemy.sql import func




class BikeStoreQueries:
    """
    Класс для выполнения запросов к базе данных BikeStore.

    Этот класс предоставляет методы для выполнения различных запросов к базе данных, 
    таких как получение списка продуктов, активных сотрудников, покупателей магазина 
    и статистики заказов.

    Атрибуты:
        session (Session): Сессия SQLAlchemy, используемая для взаимодействия с базой данных.

    Методы:
        get_products_and_brands():
            Возвращает список всех продуктов с их торговыми марками.

        get_active_employees_and_stores():
            Возвращает список всех активных сотрудников и названия магазинов, в которых они работают.

        get_customers_by_store(store_id):
            Возвращает название магазина и список всех покупателей, связанных с указанным ID магазина.

        get_product_count_by_category():
            Возвращает количество продуктов в каждой категории.

        get_order_count_per_customer():
            Возвращает количество заказов для каждого клиента.

        get_customers_with_orders():
            Возвращает список клиентов, которые сделали хотя бы один заказ, и общее количество их заказов.
    """
    
    def __init__(self, session):
        """
            нициализирует объект BikeStoreQueries.
            Args:
            session (Session): Сессия SQLAlchemy для взаимодействия с базой данных.
        """
        self.session = session

    def get_products_and_brands(self):
        """
            Возвращает список всех продуктов с их торговыми марками._
        """
        query = self.session.query(Product.product_name, Brand.brand_name).join(Brand).all()
        return query

    def get_active_employees_and_stores(self):
        """
            возвращает список всех активных сотрудников и названия магазинов, в которых они работают.
        """
        query = self.session.query(Staff.first_name, Staff.last_name, Store.store_name) \
            .join(Store).filter(Staff.active == 1).all()
        return query

    def get_customers_by_store(self, store_id):
        '''
            Возвращает название магазина и список всех покупателей, связанных с указанным ID магазина.
        '''
        # Получить название магазина по ID
        store = self.session.query(Store).filter(Store.store_id == store_id).first()
        if not store:
            return None, []

        # Получить клиентов, связанных с этим магазином
        customers = (
            self.session.query(
                Customer.first_name,
                Customer.last_name,
                Customer.email,
                Customer.phone
            )
            .join(Order, Customer.customer_id == Order.customer_id)
            .filter(Order.store_id == store_id)
            .all()
        )
        return store.store_name, customers

    def get_product_count_by_category(self):
        '''
            Возвращает количество продуктов в каждой категории.
        '''
        query = self.session.query(Category.category_name, func.count(Product.product_id).label('product_count')) \
            .join(Product).group_by(Category.category_name).all()
        return query

    def get_order_count_per_customer(self):
        '''
            Возвращает количество заказов для каждого клиента.
        '''
        query = self.session.query(Customer.first_name, Customer.last_name, func.count(Order.order_id).label('order_count')) \
            .join(Order).group_by(Customer.customer_id).all()
        return query

    def get_customers_with_orders(self):
        '''
            Возвращает список клиентов, которые сделали хотя бы один заказ, и общее количество их заказов.
        '''
        query = self.session.query(Customer.first_name, Customer.last_name, func.count(Order.order_id).label('order_count')) \
            .join(Order).group_by(Customer.customer_id).having(func.count(Order.order_id) > 0).all()
        return query

