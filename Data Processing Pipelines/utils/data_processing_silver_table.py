import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_loans(snapshot_date_str, bronze_dir, silver_dir, spark):
    # Load bronze loans data
    filepath = os.path.join(bronze_dir, f"bronze_loans_{snapshot_date_str.replace('-','_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    # Data cleaning and validation
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType()
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # Save as parquet
    output_path = os.path.join(silver_dir, f"silver_loans_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(output_path)
    
    return df

def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    # Load bronze attributes data
    filepath = os.path.join(bronze_dir, f"bronze_attributes_{snapshot_date_str.replace('-','_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    # Data cleaning and validation
    df = df.withColumn("Name", F.regexp_replace(col("Name"), "[^a-zA-Z\\s]", ""))
    df = df.withColumn("Age", 
        F.when(col("Age").rlike("^\\d+$") & col("Age").between(0, 100), col("Age").cast("int"))
         .otherwise(None))
    df = df.withColumn("SSN",
        F.when(col("SSN").rlike("^\\d{3}-\\d{2}-\\d{4}$"), col("SSN"))
         .otherwise("NA"))
    df = df.withColumn("Occupation",
        F.when(col("Occupation").rlike("^[a-zA-Z_]+$"), col("Occupation"))
         .otherwise("NA"))
    
    # Save as parquet
    output_path = os.path.join(silver_dir, f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(output_path)
    
    return df

def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    # Load bronze financials data
    filepath = os.path.join(bronze_dir, f"bronze_financials_{snapshot_date_str.replace('-','_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    # Data cleaning and validation
    # Update numeric_cols list to include all numeric columns
    numeric_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Outstanding_Debt",
        "Num_Credit_Inquiries", "Credit_Utilization_Ratio", 
        "Total_EMI_per_month", "Monthly_Balance"
    ]
    
    # Clean and convert all numeric columns to float consistently
    for col_name in numeric_cols:
        df = df.withColumn(col_name, 
            F.regexp_replace(col(col_name), "[^0-9.-]", ""))
        df = df.withColumn(col_name, 
            F.when(col(col_name) != "", col(col_name).cast("float"))
             .otherwise(None))
        df = df.withColumn(col_name, F.round(col(col_name), 2))
    
    # Range constraints for specific columns
    range_constrained_cols = [
        "Num_Credit_Card", "Num_Bank_Accounts", 
        "Interest_Rate", "Num_of_Loan"
    ]
    
    for col_name in range_constrained_cols:
        df = df.withColumn(col_name,
            F.when(col(col_name).between(0, 100), col(col_name))
             .otherwise(None))
    
    # Remove negative values for delay columns
    df = df.withColumn("Delay_from_due_date",
        F.when(col("Delay_from_due_date") >= 0, col("Delay_from_due_date"))
         .otherwise(None))
    
    df = df.withColumn("Num_of_Delayed_Payment",
        F.when(col("Num_of_Delayed_Payment") >= 0, col("Num_of_Delayed_Payment"))
         .otherwise(None))
    
    # Clean categorical columns
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "_", None)
         .otherwise(col("Credit_Mix")))
    
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", None)
         .otherwise(col("Payment_Behaviour")))
    
    # Save as parquet
    output_path = os.path.join(silver_dir, f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(output_path)
    
    return df

def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    # Load bronze clickstream data
    filepath = os.path.join(bronze_dir, f"bronze_clickstream_{snapshot_date_str.replace('-','_')}.csv")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    # Basic cleaning (add specific clickstream cleaning logic as needed)
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # Save as parquet
    output_path = os.path.join(silver_dir, f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(output_path)
    
    return df