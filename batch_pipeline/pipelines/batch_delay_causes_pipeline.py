"""
B2 Batch pipeline: "Why are flights late?" - delay-cause analysis

Reads flights, airports and airlines CSVs from GCS, finds the share of each
delay cause per airport and per airline (per month), finds each airport's main
causes, and writes the results to BigQuery.

Run (inside the spark-driver-app container):
    spark-submit --master spark://spark-master:7077 b2_delay_causes_pipeline.py \
        --project_id <project> --bucket <bucket> --dataset <bq_dataset>
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lit, to_date, upper, trim, regexp_replace, when, row_number, first,
    concat, coalesce, count,
    sum as spark_sum, max as spark_max, round as spark_round,
)

# The 5 delay-cause columns (minutes)
CAUSE_COLS = ["carrier_delay", "weather_delay", "nas_delay",
              "security_delay", "late_aircraft_delay"]

# A flight is identified by: date + airline + flight number + origin
FLIGHT_KEY = ["flight_date", "airline_code", "flight_number", "origin"]

MIN_LATE_FLIGHTS = 100   # airports with fewer late flights are left out of the summary


# ------------------------------------------------------------------
# 1 - Read and clean
# ------------------------------------------------------------------
def read_and_clean_flights(spark, path):
    raw = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(path)
    print("Raw flights:", raw.count())

    flights = raw.select(
        "flight_date", "year", "month", "airline_code", "flight_number",
        "origin", "dest", "arr_delay", "cancelled", "diverted", *CAUSE_COLS
    )

    flights = flights \
        .withColumn("flight_date",  to_date(col("flight_date"), "yyyy-MM-dd")) \
        .withColumn("airline_code", upper(trim(col("airline_code")))) \
        .withColumn("origin",       upper(trim(col("origin")))) \
        .withColumn("dest",         upper(trim(col("dest")))) \
        .withColumn("arr_delay",    col("arr_delay").cast("double")) \
        .withColumn("cancelled",    col("cancelled").cast("double")) \
        .withColumn("diverted",     col("diverted").cast("double"))

    for cause in CAUSE_COLS:
        flights = flights.withColumn(cause, col(cause).cast("double"))

    flights = flights \
        .where((col("cancelled") == 0) & (col("diverted") == 0)) \
        .dropna(subset=["flight_date", "airline_code", "dest", "arr_delay"]) \
        .na.fill(0, subset=CAUSE_COLS) \
        .dropDuplicates(FLIGHT_KEY) \
        .drop("cancelled", "diverted") \
        .withColumnRenamed("dest", "dest_airport")   # arrival airport = where arr_delay is measured

    flights.cache()
    print("Clean flights:", flights.count())
    return flights


def read_airports(spark, path):
    return spark.read.format("csv").option("header", "true").load(path) \
        .select(upper(trim(col("iata_code"))).alias("iata_code"),
                "airport_name", "city", "state") \
        .dropna(subset=["iata_code"]) \
        .dropDuplicates(["iata_code"])


def read_airlines(spark, path):
    return spark.read.format("csv").option("header", "true").load(path) \
        .select(upper(trim(col("airline_code"))).alias("airline_code"), "airline") \
        .dropna(subset=["airline_code"]) \
        .dropDuplicates(["airline_code"])


# ------------------------------------------------------------------
# 2 - Keep only late flights
# ------------------------------------------------------------------
def keep_late_flights(flights):
    late = flights.where(col("arr_delay") >= 15)
    total_cause = sum([col(cause) for cause in CAUSE_COLS], lit(0))   # Python sum over Spark columns
    late = late.where(total_cause > 0)
    print("Late flights:", late.count())
    return late


# ------------------------------------------------------------------
# 3 - Unpivot: 5 cause columns -> rows
# ------------------------------------------------------------------
def unpivot_causes(late):
    id_cols = FLIGHT_KEY + ["year", "month", "dest_airport"]

    causes_long = late.unpivot(
        ids=id_cols,
        values=CAUSE_COLS,
        variableColumnName="cause",
        valueColumnName="delay_minutes"
    )
    causes_long = causes_long \
        .where(col("delay_minutes") > 0) \
        .withColumn("cause", regexp_replace(col("cause"), "_delay$", ""))
    print("Rows after unpivot:", causes_long.count())
    return causes_long


def check_cause_totals(causes_long, late):
    """Data-quality check: do each flight's cause minutes add up to its arrival delay?"""
    check = causes_long.groupBy(FLIGHT_KEY) \
        .agg(spark_sum("delay_minutes").alias("cause_total")) \
        .join(late.select(*FLIGHT_KEY, "arr_delay"), FLIGHT_KEY)
    print("Flights where causes != arr_delay:",
          check.where(col("cause_total") != col("arr_delay")).count())


# ------------------------------------------------------------------
# 4 - Join with airports and airlines
# ------------------------------------------------------------------
def add_names(causes_long, airports, airlines):
    joined = causes_long.join(
        airports, causes_long["dest_airport"] == airports["iata_code"], "left"
    ).drop("iata_code")
    joined = joined.join(airlines, ["airline_code"], "left")
    joined = joined.na.fill("Unknown", subset=["airport_name", "city", "state", "airline"])

    print("Rows without airport match:", joined.where(col("airport_name") == "Unknown").count())
    print("Rows without airline match:", joined.where(col("airline") == "Unknown").count())
    return joined


# ------------------------------------------------------------------
# 5 - Share of each cause per airport / airline per month (window)
# ------------------------------------------------------------------
def cause_shares(joined, group_cols):
    """Share of each cause within each group (e.g. airport + year + month)."""
    result = joined.groupBy(*group_cols, "cause").agg(
        spark_sum("delay_minutes").alias("delay_minutes"),
        count("*").alias("num_late_flights")
    )
    w = Window.partitionBy(*group_cols)
    return result \
        .withColumn("total_delay_minutes", spark_sum("delay_minutes").over(w)) \
        .withColumn("cause_share", spark_round(col("delay_minutes") / col("total_delay_minutes"), 4))


# ------------------------------------------------------------------
# 6 - Each airport's main causes
# ------------------------------------------------------------------
def pct(c):
    """0.3512 -> '35'"""
    return (col(c) * 100).cast("int").cast("string")


def airport_main_causes(joined, late):
    # a. Season column
    joined_s = joined.withColumn(
        "season",
        when(col("month").isin(12, 1, 2), "Winter")
        .when(col("month").isin(3, 4, 5), "Spring")
        .when(col("month").isin(6, 7, 8), "Summer")
        .otherwise("Autumn")
    )

    # b. Top 2 causes per airport (whole period)
    airport_total = joined_s.groupBy("dest_airport", "airport_name", "city", "state", "cause") \
        .agg(spark_sum("delay_minutes").alias("delay_minutes"))
    w_ap = Window.partitionBy("dest_airport")
    airport_total = airport_total.withColumn(
        "cause_share",
        spark_round(col("delay_minutes") / spark_sum("delay_minutes").over(w_ap), 4)
    )
    w_rank = Window.partitionBy("dest_airport").orderBy(col("delay_minutes").desc(), col("cause"))
    top2 = airport_total.withColumn("rank", row_number().over(w_rank)).where(col("rank") <= 2)
    main_overall = top2.groupBy("dest_airport", "airport_name", "city", "state").agg(
        spark_max(when(col("rank") == 1, col("cause"))).alias("main_cause"),
        spark_max(when(col("rank") == 1, col("cause_share"))).alias("main_cause_share"),
        spark_max(when(col("rank") == 2, col("cause"))).alias("second_cause"),
        spark_max(when(col("rank") == 2, col("cause_share"))).alias("second_cause_share")
    )

    # c. Main cause per airport per season
    airport_season = joined_s.groupBy("dest_airport", "season", "cause") \
        .agg(spark_sum("delay_minutes").alias("delay_minutes"))
    w_aps = Window.partitionBy("dest_airport", "season")
    airport_season = airport_season.withColumn(
        "cause_share",
        spark_round(col("delay_minutes") / spark_sum("delay_minutes").over(w_aps), 4)
    )
    w_rank_s = Window.partitionBy("dest_airport", "season").orderBy(col("delay_minutes").desc(), col("cause"))
    main_season = airport_season.withColumn("rank", row_number().over(w_rank_s)).where(col("rank") == 1)

    # d. Pivot: seasons become columns
    season_wide = main_season.groupBy("dest_airport") \
        .pivot("season", ["Winter", "Spring", "Summer", "Autumn"]) \
        .agg(first("cause").alias("cause"), first("cause_share").alias("share"))

    # e. Number of late flights per airport
    late_counts = late.groupBy("dest_airport").agg(count("*").alias("num_late_flights"))

    # f. Peak season of the main cause
    peak = airport_season.join(
        main_overall.select("dest_airport", col("main_cause").alias("cause")),
        ["dest_airport", "cause"], "inner"
    )
    w_peak = Window.partitionBy("dest_airport").orderBy(col("cause_share").desc(), col("season"))
    peak = peak.withColumn("r", row_number().over(w_peak)).where(col("r") == 1) \
        .select("dest_airport",
                col("season").alias("peak_season"),
                col("cause_share").alias("peak_season_share"))

    # g. Combine into one summary row per airport
    summary = main_overall \
        .join(late_counts, ["dest_airport"], "left") \
        .join(season_wide, ["dest_airport"], "left") \
        .join(peak, ["dest_airport"], "left") \
        .where(col("num_late_flights") >= MIN_LATE_FLIGHTS) \
        .withColumn(
            "summary",
            concat(
                col("city"), lit(": mostly "), col("main_cause"),
                lit(" ("), pct("main_cause_share"), lit("%) and "),
                coalesce(col("second_cause"), lit("n/a")),
                lit(" ("), coalesce(pct("second_cause_share"), lit("-")), lit("%), especially in "),
                coalesce(col("peak_season"), lit("n/a")),
                lit(" ("), coalesce(pct("peak_season_share"), lit("-")), lit("%)")
            )
        )
    print("Airports in summary:", summary.count())
    return summary


# ------------------------------------------------------------------
# 7 - Write to BigQuery
# ------------------------------------------------------------------
def write_to_bigquery(df, table_name):
    df.write.format("bigquery") \
        .option("table", table_name) \
        .mode("overwrite") \
        .save()
    print("Written to BigQuery:", table_name)


# ------------------------------------------------------------------
# Main job
# ------------------------------------------------------------------
def run_job(project_id, bucket, dataset):
    # Section 0 - Spark session and settings
    spark = SparkSession.builder.appName("B2_DelayCauseAnalysis").getOrCreate()

    conf = spark.sparkContext._jsc.hadoopConfiguration()
    conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    conf.set("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    spark.conf.set("temporaryGcsBucket", bucket)

    flights_path  = f"gs://{bucket}/flights_sample.csv"
    airports_path = f"gs://{bucket}/airports.csv"
    airlines_path = f"gs://{bucket}/airlines.csv"

    table_airport = f"{project_id}.{dataset}.delay_causes_by_airport"
    table_airline = f"{project_id}.{dataset}.delay_causes_by_airline"
    table_summary = f"{project_id}.{dataset}.airport_main_cause"

    # Section 1
    flights  = read_and_clean_flights(spark, flights_path)
    airports = read_airports(spark, airports_path)
    airlines = read_airlines(spark, airlines_path)

    # Sections 2 and 3
    late = keep_late_flights(flights)
    causes_long = unpivot_causes(late)
    check_cause_totals(causes_long, late)

    # Section 4
    joined = add_names(causes_long, airports, airlines)

    # Section 5
    airport_month = cause_shares(
        joined, ["dest_airport", "airport_name", "city", "state", "year", "month"])
    airline_month = cause_shares(
        joined, ["airline_code", "airline", "year", "month"])

    # Section 6
    airport_summary = airport_main_causes(joined, late)

    # Section 7
    out_airport = airport_month.select(
        "dest_airport", "airport_name", "city", "state", "year", "month",
        "cause", "delay_minutes", "num_late_flights", "total_delay_minutes", "cause_share")
    out_airline = airline_month.select(
        "airline_code", "airline", "year", "month",
        "cause", "delay_minutes", "num_late_flights", "total_delay_minutes", "cause_share")
    out_summary = airport_summary.select(
        "dest_airport", "airport_name", "city", "state", "num_late_flights",
        "main_cause", "main_cause_share", "second_cause", "second_cause_share",
        "peak_season", "peak_season_share",
        "Winter_cause", "Winter_share", "Spring_cause", "Spring_share",
        "Summer_cause", "Summer_share", "Autumn_cause", "Autumn_share",
        "summary")

    write_to_bigquery(out_airport, table_airport)
    write_to_bigquery(out_airline, table_airline)
    write_to_bigquery(out_summary, table_summary)

    flights.unpersist()
    spark.stop()


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True, help="GCP project id")
    parser.add_argument("--bucket", type=str, required=True,
                        help="GCS bucket with the CSVs (also the temp bucket for BigQuery)")
    parser.add_argument("--dataset", type=str, required=True, help="BigQuery dataset for the results")
    return vars(parser.parse_args())


if __name__ == "__main__":
    run_job(**parse_command_line_arguments())