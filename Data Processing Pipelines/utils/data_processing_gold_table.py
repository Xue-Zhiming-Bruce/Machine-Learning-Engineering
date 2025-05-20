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
from functools import reduce
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loans_" + snapshot_date_str.replace('-','_') + '.parquet'  # Changed from loan_daily to loans
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # Calculate mob (months on book) from loan_start_date and snapshot_date
    df = df.withColumn("mob", 
        F.months_between(col("snapshot_date"), col("loan_start_date")).cast("int"))
    
    # get customer at mob
    df = df.filter(col("mob") == mob)

    # Calculate DPD from overdue_amt and due_amt (assuming overdue means past due)
    df = df.withColumn("dpd", 
        F.when(col("overdue_amt") > 0, 
              F.datediff(col("snapshot_date"), col("loan_start_date")) - 
              (col("installment_num") * 30))  # Consider using actual month days
         .otherwise(0))
    
    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_dirs, gold_dirs, spark):
    # Update columns to remove
    cols_to_remove = [
        'label', 'target', 'Name', 'SSN', 'Occupation',
        'Payment_of_Min_Amount', 'Payment_Behaviour', 'fe_sum'
    ]
    
    # Update numeric columns list to include all numeric features
    numeric_cols = [
        'Monthly_Balance', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Outstanding_Debt', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
        'Total_EMI_per_month', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment'
    ]
    
    # Initialize feature_dfs list before using it
    feature_dfs = []
    
    # Load and process financial features
    financial_path = os.path.join(silver_dirs["financials"], 
                               f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(financial_path):
        financial_df = spark.read.parquet(financial_path)
        feature_dfs.append(financial_df)
    
    # Load and process attribute features
    attribute_path = os.path.join(silver_dirs["attributes"], 
                                f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(attribute_path):
        attribute_df = spark.read.parquet(attribute_path)
        feature_dfs.append(attribute_df)
    
    # Load and process clickstream features
    clickstream_path = os.path.join(silver_dirs["clickstream"], 
                                  f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    if os.path.exists(clickstream_path):
        clickstream_df = spark.read.parquet(clickstream_path)
        feature_dfs.append(clickstream_df)
    
    if not feature_dfs:
        print("Error: No feature files found")
        return None
    
    # Change join type to preserve more data
    features_df = reduce(lambda a, b: a.join(b, ["Customer_ID", "snapshot_date"], "left"), feature_dfs)
    
    if features_df.count() == 0:
        print("Error: Empty DataFrame loaded")
        return None

    # Encode Occupation as integer
    if "Occupation" in features_df.columns:
        # Create occupation mapping
        occupation_dict = features_df.select("Occupation").distinct().na.drop() \
            .rdd.map(lambda x: x[0]).zipWithIndex() \
            .collectAsMap()
        
        # Broadcast the mapping
        bc_occupation_dict = spark.sparkContext.broadcast(occupation_dict)
        
        # Define UDF for encoding
        def encode_occupation(occupation):
            return bc_occupation_dict.value.get(occupation)
        
        encode_udf = F.udf(encode_occupation, IntegerType())
        
        # Apply encoding
        features_df = features_df.withColumn("Occupation_encoded", 
            encode_udf(F.col("Occupation")))

    # Save gold feature store table
    gold_path = gold_dirs["feature_store"] + f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    features_df.write.mode("overwrite").parquet(gold_path)
    print(f"Gold feature store saved to: {gold_path}, row count: {features_df.count()}")

    # Final check to ensure no label columns remain
    final_cols = [c for c in features_df.columns 
                 if not c.lower().startswith(('label', 'target')) 
                 and c not in cols_to_remove]
    features_df = features_df.select(*final_cols)
    
    # After merging all feature DataFrames, add type casting
    # Add NULL handling before type casting
    for col_name in numeric_cols:
        if col_name in features_df.columns:
            features_df = features_df.withColumn(col_name, 
                F.when(col(col_name).isNotNull(), col(col_name))
                 .otherwise(F.lit(None).cast(FloatType())))  # Explicit null handling
    
    # Add Credit_Mix encoding
    if "Credit_Mix" in features_df.columns:
        credit_mix_mapping = {
            "Good": 2,
            "Standard": 1,
            "Bad": 0
        }
        features_df = features_df.withColumn("Credit_Mix_encoded",
            F.when(col("Credit_Mix") == "Good", credit_mix_mapping["Good"])
             .when(col("Credit_Mix") == "Standard", credit_mix_mapping["Standard"])
             .otherwise(credit_mix_mapping["Bad"]))
    
    return features_df