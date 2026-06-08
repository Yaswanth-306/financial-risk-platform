from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os
from dotenv import load_dotenv

load_dotenv('/home/chitr/financial-risk-platform/.env')

# Initialize Spark
spark = SparkSession.builder \
    .appName("FinancialRiskFeatureEngineering") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Database connection properties
DB_URL = f"jdbc:postgresql://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
DB_PROPERTIES = {
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "driver": "org.postgresql.Driver"
}

print("Reading raw stock prices from PostgreSQL...")
df = spark.read.jdbc(
    url=DB_URL,
    table="raw_stock_prices",
    properties=DB_PROPERTIES
)

print(f"Loaded {df.count()} rows")
df.show(5)

# Define window specs
window_ticker = Window.partitionBy("ticker").orderBy("date")
window_7d  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-6, 0)
window_14d = Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
window_30d = Window.partitionBy("ticker").orderBy("date").rowsBetween(-29, 0)

print("Engineering features...")

df_features = df \
    .withColumn("prev_close", F.lag("close", 1).over(window_ticker)) \
    .withColumn("daily_return", (F.col("close") - F.col("prev_close")) / F.col("prev_close")) \
    .withColumn("ma_7",  F.avg("close").over(window_7d)) \
    .withColumn("ma_14", F.avg("close").over(window_14d)) \
    .withColumn("ma_30", F.avg("close").over(window_30d)) \
    .withColumn("volatility_7d",  F.stddev("daily_return").over(window_7d)) \
    .withColumn("volatility_30d", F.stddev("daily_return").over(window_30d)) \
    .withColumn("high_low_range", (F.col("high") - F.col("low")) / F.col("low")) \
    .withColumn("volume_ma_7", F.avg("volume").over(window_7d)) \
    .withColumn("volume_ratio", F.col("volume") / F.col("volume_ma_7")) \
    .withColumn("price_vs_ma30", (F.col("close") - F.col("ma_30")) / F.col("ma_30")) \
    .withColumn("risk_label", 
        F.when(F.col("volatility_30d") > 0.02, 1).otherwise(0)
    ) \
    .dropna()

print(f"Features engineered: {df_features.count()} rows")
df_features.show(5)

# Save features back to PostgreSQL
print("Saving features to PostgreSQL...")
df_features.select(
    "ticker", "date", "close", "daily_return",
    "ma_7", "ma_14", "ma_30",
    "volatility_7d", "volatility_30d",
    "high_low_range", "volume_ratio",
    "price_vs_ma30", "risk_label"
).write.jdbc(
    url=DB_URL,
    table="stock_features",
    mode="overwrite",
    properties=DB_PROPERTIES
)

print("✅ Feature engineering complete! Data saved to stock_features table")
spark.stop()
