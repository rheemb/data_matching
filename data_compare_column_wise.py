from pyspark.sql import SparkSession
from pyspark.sql.functions import col, md5, concat_ws, when
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("FullColumnMatching").getOrCreate()

# Load data
df1 = spark.read.option("header", "true").csv("source_file.csv")
df2 = spark.read.option("header", "true").csv("target_file.csv")

# Rename columns
column_mapping = {
    "cd": "system_cd", "acer-status": "accr_cd", "accr-status-desc": "accr_status_desc",
    "notetsrt": "note_txt", "start-date": "start_date", "end-date": "end_date",
    "status": "status", "last-upd-user-id": "last_upd_user_id", "last-upd-dt": "last_upd_dt"
}

for old_col, new_col in column_mapping.items():
    df1 = df1.withColumnRenamed(old_col, new_col)
    df2 = df2.withColumnRenamed(old_col, new_col)

all_columns = ["system_cd", "accr_cd", "accr_status_desc", "note_txt", 
               "start_date", "end_date", "status", "last_upd_user_id", "last_upd_dt"]

# Add hash for each record
df1 = df1.withColumn("full_hash", md5(concat_ws("|", *[col(c) for c in all_columns])))
df2 = df2.withColumn("full_hash", md5(concat_ws("|", *[col(c) for c in all_columns])))

# Add blocking key based on multiple columns for better partitioning
df1 = df1.withColumn("block_key", concat_ws("_", col("system_cd"), F.substring(col("accr_cd"), 1, 2)))
df2 = df2.withColumn("block_key", concat_ws("_", col("system_cd"), F.substring(col("accr_cd"), 1, 2)))

# FULL JOIN to compare ALL columns side by side
full_comparison = df1.alias("src").join(
    df2.alias("tgt"),
    ["system_cd", "accr_cd"],  # Join on key columns
    "full_outer"
)

# Add column-wise comparison for EVERY column
comparison_df = full_comparison
for column in all_columns:
    comparison_df = comparison_df.withColumn(
        f"{column}_match",
        F.coalesce(F.col(f"src.{column}") == F.col(f"tgt.{column}"), F.lit(False))
    )

# Add overall match status
comparison_df = comparison_df.withColumn(
    "all_columns_match",
    F.col("system_cd_match") & F.col("accr_cd_match") & F.col("accr_status_desc_match") &
    F.col("note_txt_match") & F.col("start_date_match") & F.col("end_date_match") &
    F.col("status_match") & F.col("last_upd_user_id_match") & F.col("last_upd_dt_match")
)

# Categorize the records
result_df = comparison_df.withColumn(
    "match_status",
    F.when(F.col("src.system_cd").isNull(), "ONLY_IN_TARGET")
     .when(F.col("tgt.system_cd").isNull(), "ONLY_IN_SOURCE")
     .when(F.col("all_columns_match"), "EXACT_MATCH")
     .otherwise("FUZZY_MATCH")  # Same keys but different other columns
)

# Show detailed comparison for fuzzy matches
fuzzy_matches = result_df.filter(F.col("match_status") == "FUZZY_MATCH")

print("=== DETAILED COLUMN-WISE COMPARISON ===")
print(f"Total records with same keys but different values: {fuzzy_matches.count()}")

# Show exactly which columns differ for each fuzzy match
for column in all_columns:
    fuzzy_matches = fuzzy_matches.withColumn(
        f"{column}_diff",
        F.when(F.col(f"src.{column}") != F.col(f"tgt.{column}"), 
              F.concat(F.col(f"src.{column}"), F.lit(" ≠ "), F.col(f"tgt.{column}")))
         .otherwise(F.lit("✓"))
    )

# Display the differences
fuzzy_matches.select(
    "system_cd", "accr_cd", 
    *[f"{col}_diff" for col in all_columns],
    "match_status"
).show(truncate=False)

# Show specific example: SPL* vs SPL
print("\n=== SPECIFIC EXAMPLE: SPL* vs SPL ===")
spl_example = fuzzy_matches.filter(
    (F.col("src.accr_cd").contains("SPL")) | (F.col("tgt.accr_cd").contains("SPL"))
).select(
    "src.accr_cd", "tgt.accr_cd", 
    "src.accr_status_desc", "tgt.accr_status_desc",
    "src.note_txt", "tgt.note_txt",
    "match_status"
)

spl_example.show(truncate=False)

# Summary statistics
print("\n=== MATCHING SUMMARY ===")
result_df.groupBy("match_status").count().show()

# Count mismatches per column
print("\n=== COLUMN MISMATCH COUNT ===")
mismatch_counts = []
for column in all_columns:
    count = fuzzy_matches.filter(F.col(f"{column}_match") == False).count()
    mismatch_counts.append((column, count))

mismatch_df = spark.createDataFrame(mismatch_counts, ["column", "mismatch_count"])
mismatch_df.show()

spark.stop()
