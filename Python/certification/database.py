from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Подключение к БД
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/sakila"

# Создание движка SQLAlchemy
engine = create_engine(DATABASE_URL)

# Создание сессии для работы с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Функция для получения сессии
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
