from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("PostgreSQL Integration") \
#     .config("spark.jars", "/home/maksim/.local/share/DBeaverData/drivers/maven/maven-central/org.postgresql/postgresql-42.7.2.jar") \
#     .getOrCreate()

def get_spark_session():
    """
    Создаёт и возвращает объект SparkSession для подключения к базе данных.

    Returns:
        SparkSession: Настроенная SparkSession.
    """
    spark = SparkSession.builder \
        .appName("BikeStore Queries") \
        .config("spark.jars", "/home/maksim/Документы/libs/postgresql-42.7.4.jar") \
        .getOrCreate()
    return spark



if __name__ == "__main__":

    # Пример запроса с использованием JDBC
    jdbc_url = "jdbc:postgresql://localhost:5432/BikeStore"
    properties = {
        "user": "postgres",
        "password": "postgres",
        "driver": "org.postgresql.Driver"
    }
    
    spark = get_spark_session()
    # Чтение данных с использованием JDBC
    df = spark.read.jdbc(jdbc_url, "products", properties=properties)
    df.show()
