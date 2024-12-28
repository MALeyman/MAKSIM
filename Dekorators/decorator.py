# -*- coding: utf-8 -*-
"""
Создано: 28.12.2024

@Автор: Лейман М.А.
"""

# Определение глобальной переменной user_role
user_role = ""

def role_required(role: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if user_role == role:
                return func(*args, **kwargs)
            else:
                return "Permission denied"
        return wrapper
    return decorator

@role_required("admin")
def secret_resource() -> str:
    return "Permission accepted"

# Пример использования
if __name__ == "__main__":
    # Пример 1: Пользователь с ролью admin
    user_role = "admin"
    print(secret_resource())  # Ожидаемый вывод: Permission accepted

    # Пример 2: Пользователь с ролью manager
    user_role = "manager"
    print(secret_resource())  # Ожидаемый вывод: Permission denied
