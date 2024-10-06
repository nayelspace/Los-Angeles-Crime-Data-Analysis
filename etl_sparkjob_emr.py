from itertools import chain
import sys

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col, when, lit, concat, row_number, lpad, dayofmonth, month, year, to_date, date_format, substring, to_utc_timestamp, udf, create_map, from_unixtime, unix_timestamp
from pyspark.sql.types import StructType, FloatType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType

from pyspark.sql import SparkSession

# EMR cluster ID and endpoint
cluster_id = ""
emr_endpoint = ""
aws_access_key = ""
aws_secret_key = ""

# Create Spark session
spark = SparkSession.builder \
    .appName("Data Processing") \
    .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
    .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") \
    .getOrCreate()

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

glueContext = GlueContext(spark.sparkContext)
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


schema = StructType([
    StructField("DR_NO", StringType(), True),
    StructField("Date Rptd", StringType(), True),
    StructField("DATE OCC", StringType(), True),
    StructField("TIME OCC", StringType(), True),
    StructField("AREA", IntegerType(), True),
    StructField("AREA NAME", StringType(), True),
    StructField("Rpt Dist No", StringType(), True),
    StructField("Part 1-2", StringType(), True),
    StructField("Crm Cd", IntegerType(), True),
    StructField("Crm Cd Desc", StringType(), True),
    StructField("Mocodes", StringType(), True),
    StructField("Vict Age", IntegerType(), True),
    StructField("Vict Sex", StringType(), True),
    StructField("Vict Descent", StringType(), True),
    StructField("Premis Cd", IntegerType(), True),
    StructField("Premis Desc", StringType(), True),
    StructField("Weapon Used Cd", IntegerType(), True),
    StructField("Weapon Desc", StringType(), True),
    StructField("Status", StringType(), True),
    StructField("Status Desc", StringType(), True),
    StructField("Crm Cd 1", IntegerType(), True),
    StructField("Crm Cd 2", IntegerType(), True),
    StructField("Crm Cd 3", IntegerType(), True),
    StructField("Crm Cd 4", IntegerType(), True),
    StructField("LOCATION", StringType(), True),
    StructField("Cross Street", StringType(), True),
    StructField("LAT", DoubleType(), True),
    StructField("LON", DoubleType(), True)
])


# Read CSV file
df = spark.read.csv("s3://pipdumpdatabucket/raw_dataset/rawdataset.csv", header=True, schema=schema)

# Drop unwanted columns
df = df.drop('DR_NO', 'Rpt Dist No', 'Part 1-2', 'Mocodes',
             'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street')

# Convert columns to appropriate data types
df = df.withColumn("Date Rptd", date_format(to_date(substring(df["Date Rptd"], 1, 10), 'MM/dd/yyyy'), 'yyyy-MM-dd'))
df = df.withColumn("DATE OCC", date_format(to_date(substring(df["DATE OCC"], 1, 10), 'MM/dd/yyyy'), 'yyyy-MM-dd'))

# Pad 'TIME OCC' with zeros and add colon
@udf(returnType=StringType())
def pad_time(time_str):
    return time_str.zfill(4)[:2] + ':' + time_str.zfill(4)[2:]

df = df.withColumn("TIME OCC", pad_time("TIME OCC"))

# Fix 'Vict Age' values
df = df.withColumn("Vict Age", when(col("Vict Age") <= 0, None).otherwise(col("Vict Age")))

# Fix 'Vict Sex' values
df = df.withColumn("Vict Sex", when(col("Vict Sex").isin(['H', 'N', '-']), 'X').otherwise(col("Vict Sex")))
df = df.fillna("N/A", subset=["Vict Sex"])

# Fix 'Vict Descent' values
df = df.fillna("N/A", subset=["Vict Descent"])

vict_descent_codes = {
    'A': 'Other Asian',
    'B': 'Black',
    'C': 'Chinese',
    'D': 'Cambodian',
    'F': 'Filipino',
    'G': 'Guamanian',
    'H': 'Hispanic/Latin/Mexican',
    'I': 'American Indian/Alaskan Native',
    'J': 'Japanese',
    'K': 'Korean',
    'L': 'Laotian',
    'N/A': 'N/A',
    'O': 'Other',
    'P': 'Pacific Islander',
    'S': 'Samoan',
    'U': 'Hawaiian',
    'V': 'Vietnamese',
    'W': 'White',
    'X': 'Unknown',
    'Z': 'Asian Indian',
    '-': 'Unknown'
}

@udf(returnType=StringType())
def map_vict_descent(code):
    return vict_descent_codes.get(code, None)

df = df.withColumn("Vict Descent", map_vict_descent("Vict Descent"))

# Fix 'Premis Cd' and 'Premis Desc' values
df = df.fillna(0, subset=["Premis Cd"])
df = df.fillna("UNKNOWN", subset=["Premis Desc"])

# Add a new column 'row_number' with the row number for each row
df_with_row_number = df.withColumn('row_number', row_number().over(Window.orderBy('Weapon Used Cd')) - 1)

# Filter rows where column 'Weapon Used Cd' is equal to 222
filtered_rows = df_with_row_number.filter(col('Weapon Used Cd') == 222)

# Extract 'row_number' column values as a list
indices = [row['row_number'] for row in filtered_rows.collect()]

# Filter the DataFrame to exclude the rows with the specified indices
df_without_rows = df_with_row_number.filter(~col('row_number').isin(indices))

# Drop the 'row_number' column to return the DataFrame to its original form
df = df_without_rows.drop('row_number')  # Overwrite the original DataFrame

# Fix 'Weapon Desc' and 'Weapon Used Cd' values
df = df.fillna("N/A", subset=["Weapon Desc"])
df = df.fillna(0, subset=["Weapon Used Cd"])

# Fix 'Status Desc' and 'Status' values
df = df.withColumn("Status", when(col("Status Desc") == "UNK", "CC").otherwise(col("Status")))

# Excluding records with a 0 LAT and 0 LON
df = df.filter((col("LAT") != 0) | (col("LON") != 0))

df = df.dropDuplicates()

# Replace spaces with underscores in column names for parquet format
for col_name in df.columns:
    df = df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

# Write the DataFrame to Parquet file
df.repartition(8).write.mode('overwrite').parquet('s3://pipdumpdatabucket/processed_data/')

# Complete the job
job.commit()
