"""
Создано: 31.12.2024

@Автор: Лейман М.А.
"""
'''
    Классы моделей таблиц
'''

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, Date, DECIMAL, func

from sqlalchemy import Column, Integer, String, ForeignKey, DECIMAL, Boolean, Date
from sqlalchemy.orm import declarative_base, relationship


# Создание базового класса для моделей
Base = declarative_base()

class Store(Base):
    __tablename__ = 'stores'
    store_id = Column(Integer, primary_key=True)
    store_name = Column(String, nullable=False)
    phone = Column(String)
    email = Column(String)
    street = Column(String)
    city = Column(String)
    state = Column(String)
    zip_code = Column(String)

    staffs = relationship('Staff', back_populates='store')
    orders = relationship('Order', back_populates='store')
    stocks = relationship('Stock', back_populates='store')
    
    
    
class Brand(Base):
    __tablename__ = 'brands'
    brand_id = Column(Integer, primary_key=True)
    brand_name = Column(String, nullable=False)

    products = relationship('Product', back_populates='brand')
    

class Category(Base):
    __tablename__ = 'categories'
    category_id = Column(Integer, primary_key=True)
    category_name = Column(String, nullable=False)

    products = relationship('Product', back_populates='category')
    

class Product(Base):
    __tablename__ = 'products'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String, nullable=False)
    brand_id = Column(Integer, ForeignKey('brands.brand_id'))
    category_id = Column(Integer, ForeignKey('categories.category_id'))
    model_year = Column(Integer)
    list_price = Column(DECIMAL)

    brand = relationship('Brand', back_populates='products')
    category = relationship('Category', back_populates='products')
    stocks = relationship('Stock', back_populates='product')
    order_items = relationship('OrderItem', back_populates='product')
    

class Customer(Base):
    __tablename__ = 'customers'
    customer_id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    phone = Column(String)
    email = Column(String)
    street = Column(String)
    city = Column(String)
    state = Column(String)
    zip_code = Column(String)

    orders = relationship('Order', back_populates='customer')
    

class Order(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'))
    order_status = Column(String)
    order_date = Column(Date)
    required_date = Column(Date)
    shipped_date = Column(Date)
    store_id = Column(Integer, ForeignKey('stores.store_id'))
    staff_id = Column(Integer, ForeignKey('staffs.staff_id'))

    customer = relationship('Customer', back_populates='orders')
    store = relationship('Store', back_populates='orders')
    staff = relationship('Staff', back_populates='orders')
    order_items = relationship('OrderItem', back_populates='order')
    

class OrderItem(Base):
    __tablename__ = 'order_items'
    order_id = Column(Integer, ForeignKey('orders.order_id'), primary_key=True)
    item_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.product_id'))
    quantity = Column(Integer)
    list_price = Column(DECIMAL)
    discount = Column(DECIMAL)

    order = relationship('Order', back_populates='order_items')
    product = relationship('Product', back_populates='order_items')
    

class Stock(Base):
    __tablename__ = 'stocks'
    store_id = Column(Integer, ForeignKey('stores.store_id'), primary_key=True)
    product_id = Column(Integer, ForeignKey('products.product_id'), primary_key=True)
    quantity = Column(Integer)

    store = relationship('Store', back_populates='stocks')
    product = relationship('Product', back_populates='stocks')
    

class Staff(Base):
    __tablename__ = 'staffs'
    staff_id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String)
    phone = Column(String)
    active = Column(Integer)
    store_id = Column(Integer, ForeignKey('stores.store_id'))
    manager_id = Column(Integer, ForeignKey('staffs.staff_id'))

    store = relationship('Store', back_populates='staffs')
    orders = relationship('Order', back_populates='staff')
    manager = relationship('Staff', remote_side=[staff_id])
    


    
