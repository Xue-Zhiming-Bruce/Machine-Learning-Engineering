import pandas as pd
import os
import pyspark
import datetime
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
from utils.date_utils import generate_first_of_month_dates
import shutil
import glob

def save_single_parquet(df, tmp_dir, final_path):
    df.repartition(1).write.mode("overwrite").parquet(tmp_dir)
    part_file = glob.glob(os.path.join(tmp_dir, 'part-*.parquet'))[0]
    shutil.move(part_file, final_path)
    shutil.rmtree(tmp_dir)

# Load raw data
clickstream = pd.read_csv('data/feature_clickstream.csv')
attributes = pd.read_csv('data/features_attributes.csv')
financials = pd.read_csv('data/features_financials.csv')
loans = pd.read_csv('data/lms_loan_daily.csv')

# Bronze tables
os.makedirs('datamart/bronze', exist_ok=True)


# Bronze tables
# Initialize SparkSession first
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Generate dates before using them
snapshot_date_str = "2023-01-01"
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Then proceed with bronze tables processing
os.makedirs('datamart/bronze/clickstream', exist_ok=True)
os.makedirs('datamart/bronze/attributes', exist_ok=True)
os.makedirs('datamart/bronze/financials', exist_ok=True)
os.makedirs('datamart/bronze/loans', exist_ok=True)

for date_str in dates_str_lst:
    clickstream.to_csv(f'datamart/bronze/clickstream/bronze_clickstream_{date_str.replace("-","_")}.csv', index=False)
    attributes.to_csv(f'datamart/bronze/attributes/bronze_attributes_{date_str.replace("-","_")}.csv', index=False)
    financials.to_csv(f'datamart/bronze/financials/bronze_financials_{date_str.replace("-","_")}.csv', index=False)
    # Filter loans data by snapshot_date before saving
    loans_filtered = loans[loans['snapshot_date'] == date_str]
    loans_filtered.to_csv(f'datamart/bronze/loans/bronze_loans_{date_str.replace("-","_")}.csv', index=False)

# Silver: Clean and join data
clickstream_clean = clickstream.drop_duplicates().fillna(0)
attributes_clean = attributes.drop_duplicates().fillna({'Occupation': 'Unknown'})
financials_clean = financials.drop_duplicates().fillna(0)

# Join tables on Customer_ID and snapshot_date
features = pd.merge(clickstream_clean, attributes_clean, on=['Customer_ID', 'snapshot_date'], how='left')
features = pd.merge(features, financials_clean, on=['Customer_ID', 'snapshot_date'], how='left')

# Save silver table
os.makedirs('datamart/silver', exist_ok=True)
features.to_csv('datamart/silver/features.csv', index=False)

# Gold: Final feature engineering
# Example: create an aggregate feature and encode a categorical
if 'fe_1' in features.columns and 'fe_2' in features.columns:
    features['fe_sum'] = features[[col for col in features.columns if col.startswith('fe_')]].sum(axis=1)
if 'Occupation' in features.columns:
    features['Occupation_encoded'] = features['Occupation'].astype('category').cat.codes

# Save gold table
os.makedirs('datamart/gold', exist_ok=True)
features.to_csv('datamart/gold/feature_store.csv', index=False)

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

snapshot_date_str = "2023-01-01"
start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Bronze directories
bronze_dirs = {
    "clickstream": "datamart/bronze/clickstream/",
    "attributes": "datamart/bronze/attributes/",
    "financials": "datamart/bronze/financials/",
    "loans": "datamart/bronze/loans/"
}
for d in bronze_dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# Silver directories
silver_dirs = {
    "clickstream": "datamart/silver/clickstream/",
    "attributes": "datamart/silver/attributes/",
    "financials": "datamart/silver/financials/",
    "loans": "datamart/silver/loans/"
}
for d in silver_dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# Gold directories
gold_dirs = {
    "feature_store": "datamart/gold/feature_store/",
    "label_store": "datamart/gold/label_store/"
}
for d in gold_dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# Process bronze tables for each date
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_dirs, spark)

# Then process silver tables for each date
for date_str in dates_str_lst:
    # Process each dataset separately
    utils.data_processing_silver_table.process_silver_clickstream(
        date_str,
        bronze_dirs["clickstream"],
        silver_dirs["clickstream"],
        spark
    )
    
    utils.data_processing_silver_table.process_silver_attributes(
        date_str,
        bronze_dirs["attributes"],
        silver_dirs["attributes"],
        spark
    )
    
    utils.data_processing_silver_table.process_silver_financials(
        date_str,
        bronze_dirs["financials"],
        silver_dirs["financials"],
        spark
    )
    
    utils.data_processing_silver_table.process_silver_loans(
        date_str,
        bronze_dirs["loans"],
        silver_dirs["loans"],
        spark
    )

# Process gold feature store for each date (feature engineering)
# Process gold feature store for each date
for date_str in dates_str_lst:
    feature_df = utils.data_processing_gold_table.process_features_gold_table(
        date_str, 
        silver_dirs, 
        gold_dirs,
        spark
    )
    if feature_df is not None:
        tmp_dir = gold_dirs["feature_store"] + f"tmp_gold_feature_store_{date_str.replace('-','_')}"
        final_file = gold_dirs["feature_store"] + f"gold_feature_store_{date_str.replace('-','_')}.parquet"
        save_single_parquet(feature_df, tmp_dir, final_file)
        print(f"Saved gold features to: {final_file}")  # Added logging

# Process gold label store for each date
for date_str in dates_str_lst:
    label_df = utils.data_processing_gold_table.process_labels_gold_table(
        date_str, 
        silver_dirs["loans"],  # Changed from "loan_daily" to "loans"
        gold_dirs["label_store"], 
        spark, 
        dpd=30, 
        mob=6
    )
    if label_df is not None:
        tmp_dir = gold_dirs["label_store"] + f"tmp_gold_label_store_{date_str.replace('-','_')}"
        final_file = gold_dirs["label_store"] + f"gold_label_store_{date_str.replace('-','_')}.parquet"
        save_single_parquet(label_df, tmp_dir, final_file)


# Example: Load and show gold feature store
folder_path = gold_dirs["feature_store"]
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:", df.count())
df.show()