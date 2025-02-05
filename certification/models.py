from sqlalchemy import Column, Integer, String, ForeignKey, Numeric, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Таблица "Фильмы"
class Film(Base):
    __tablename__ = "film"
    
    film_id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    rating = Column(String)
    rental_duration = Column(Integer)
    replacement_cost = Column(Numeric)

# Таблица "Категории фильмов"
class Category(Base):
    __tablename__ = "category"
    
    category_id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

# Таблица "Связь фильмов с категориями"
class FilmCategory(Base):
    __tablename__ = "film_category"

    film_id = Column(Integer, ForeignKey("film.film_id"), primary_key=True)
    category_id = Column(Integer, ForeignKey("category.category_id"), primary_key=True)

# Таблица "Аренда"
class Rental(Base):
    __tablename__ = "rental"

    rental_id = Column(Integer, primary_key=True, index=True)
    rental_date = Column(DateTime)
    inventory_id = Column(Integer)
    customer_id = Column(Integer)
