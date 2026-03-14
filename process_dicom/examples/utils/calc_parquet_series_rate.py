from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, broadcast, count, broadcast, avg
from pyspark.sql.types import *
import pyspark.sql.functions as F

from dotenv import load_dotenv
import os

load_dotenv()

# Configuration constants
POSTGRES_HOST = os.getenv('host')
POSTGRES_PORT = os.getenv('port')
POSTGRES_DB = os.getenv('dbname')
POSTGRES_USER = os.getenv('user')
POSTGRES_PASSWORD = os.getenv('password')
POSTGRES_TABLE = os.getenv('POSTGRES_TABLE', 'your_schema.dicom_cdm')


def create_spark_session(app_name="DicomTagProcessing"):
    """
    Spark 세션 생성 및 설정
    """
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "16g")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "100")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "16g")
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")
            .getOrCreate())

def extract_unique_tags(spark, parquet_path, output_path):
    """
    Category와 source_modality별 고유한 태그 목록 추출
    
    Args:
        spark: SparkSession
        parquet_path: Parquet 파일 경로
        output_path: 결과 저장 경로
    """
    # Parquet 파일 로드
    tag_df = spark.read.parquet(parquet_path).withColumn("patient_no", col("patient_no").cast("string"))
    
    cohort = spark.read.option("header", "true")\
                          .option("inferSchema", "false")\
                          .csv('./curation_cohort.csv')\
                          .select("person_source_value", "category")\
                          .withColumn("person_source_value", col("person_source_value").cast("string"))
                          
    df = tag_df.join(
        broadcast(cohort),
        col("patient_no") == col("person_source_value"),
        "inner"
    )             
    # Category와 source_modality별 고유한 태그 조합 추출
    unique_tags = (df
        .select("category", "source_modality", "tag_sequence", "tag_name")
        .distinct()
        .cache())
    
    # 결과 저장
    (unique_tags
     .coalesce(1)  # 단일 파티션으로 만듦
     .write
     .mode("overwrite")
     .csv(f"{output_path}/unique_tags.csv", header=True))
    
    return unique_tags


def process_tag_statistics(spark, parquet_path, output_path):
    """
    태그별 시리즈 통계 계산
    """
    # PostgreSQL 데이터 로드
    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    connection_properties = {
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "driver": "org.postgresql.Driver",
        "fetchsize": "10000"  # 페치 사이즈 설정
    }
    filtered_table_query = f"""
        (WITH patient AS (
            SELECT DISTINCT
                person_source_value ,
                'cohort_a' AS category
            FROM
                your_schema.cohort_a ccbd
            UNION ALL
            SELECT DISTINCT
                person_source_value ,
                'cohort_b' AS category
            FROM
                your_schema.cohort_b ccbm
        )
        SELECT distinct
            category ,
            source_modality, 
            person_source_value,
            seriesinstanceuid ,
            sum(first_series) OVER (PARTITION BY category, source_modality) AS total_series
        FROM(
            SELECT 
                category,
                source_modality, 
                person_source_value ,
                seriesinstanceuid ,
                CASE
                    WHEN row_number() over(PARTITION BY category, source_modality, seriesinstanceuid) = 1
                    THEN 1 
                ELSE 0
                END AS first_series
            FROM
                curation_cdm.dicom_key_cdm dkc 
                INNER JOIN 
                    patient b
                    ON dkc.patient_no = b.person_source_value
            WHERE
                rename_filepath IS NOT NULL
            ) a
        ) a
        """
    
    # Parquet 파일 로드 및 seriesinstanceuid 생성
    parquet_df = spark.read.parquet(parquet_path)
    print(parquet_df.columns)
    parquet_df = parquet_df.select("patient_no", "seriesinstanceuid", "tag_sequence", "tag_name", "tag_value", "source_filepath", "source_modality")\
                            .withColumn("patient_no", col("patient_no").cast("string"))
    
    # PostgreSQL 데이터 로드 (파티셔닝 적용)
    postgres_df = spark.read.jdbc(
        url=jdbc_url,
        table=filtered_table_query,
        # column="source_modality",
        # lowerBound=1,
        # upperBound=2000000,
        # numPartitions=50,  # 파티션 수 증가
        properties=connection_properties
    )
    
    postgres_df = postgres_df.select(
            [col(c).alias(f"postgres_{c}") if c in ["patient_no", "seriesinstanceuid", "source_modality"] else col(c) 
             for c in postgres_df.columns]
        )
    postgres_df = postgres_df.withColumn("person_source_value", col("person_source_value").cast("string"))
    
    df_with_series = parquet_df.join(
        postgres_df,
        (parquet_df.patient_no == postgres_df.person_source_value) & (parquet_df.seriesinstanceuid == postgres_df.postgres_seriesinstanceuid),
        "inner"
    )
    print(df_with_series.count())
    
    # 전체 시리즈 수 계산 (분모로 사용)
    total_series = df_with_series.select("category", "source_modality", "seriesinstanceuid") \
        .distinct() \
        .groupBy("category", "source_modality") \
        .agg(F.count("seriesinstanceuid").alias("total_series_count"))
        
    
    # 태그별 시리즈 수 계산
    tag_counts = df_with_series.select(
        "category",
        "source_modality",
        "tag_sequence",
        "tag_name",
        "seriesinstanceuid"
    ).distinct() \
    .groupBy("category", "source_modality", "tag_sequence", "tag_name") \
    .agg(F.count("seriesinstanceuid").alias("tag_series_count"))
        
    # 태그 보유율 계산
    result = tag_counts.join(
        total_series,
        ["category", "source_modality"]
    ).withColumn(
        "retention_rate",
        F.round(F.col("tag_series_count") / F.col("total_series_count") * 100, 2)
    )
    # .select(
    #     "category",
    #     "source_modality",
    #     "tag_sequence",
    #     "tag_name",
    #     "series_count",
    #     "total_series_count",  # 전체 시리즈 수 추가
    #     "retention_rate"
    # )
    
    # 결과 정렬
    final_result = result.orderBy(
        "category",
        "source_modality",
        "tag_sequence",
        "tag_name"
    )
    print(final_result.show(10))
    
    
    # # category와 source_modality 기준으로 태그 분석
    # tag_stats = df_with_series.groupBy(
    #     "category", 
    #     "source_modality",
    #     "tag_sequence",
    #     "tag_name"
    # ).agg(
    #     count("series_id").alias("series_count"),
    #     F.countDistinct("series_id").alias("unique_series_count")
    # )
    # print(tag_stats.show(20))
    
    return final_result

def save_results(df, output_path):
    # 결과를 CSV 형태로 저장
    df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)
        
        
def main():
    # Spark 세션 생성
    spark = create_spark_session()
    
    # 경로 설정
    parquet_path = "./output/tag_info_temp/*.parquet"
    output_base_path = "./result/"
    
    try:
        # 1단계: 고유한 태그 목록 추출
        print("Extracting unique tags...")
        # unique_tags_df = extract_unique_tags(
        #     spark=spark,
        #     parquet_path=parquet_path,
        #     output_path=output_base_path
        # )
        
        # # 결과 샘플 출력
        # print("\nUnique Tags Sample:")
        # unique_tags_df.show(5)
        
        # 2. 태그 통계 계산
        print("Calculating tag statistics...")
        tag_stats = process_tag_statistics(
            spark=spark,
            parquet_path=parquet_path,
            output_path=output_base_path
        )
        
        # 결과 확인 (샘플)
        tag_stats.show(truncate=False)
        
        # 결과 저장
        save_results(tag_stats, output_path = output_base_path + 'tag_series_counts_statistics.csv')
        
        # # 결과 샘플 출력
        # print("\nTag Statistics Sample:")
        # tag_stats.orderBy("category", "source_modality", "tag_sequence", "tag_name").show(5)
        
        # # 요약 통계 출력
        # print("\nSummary by Category and Modality:")
        # tag_stats.groupBy("category", "source_modality", "tag_sequence", "tag_name").agg(
        #     count("*").alias("total_tags"),
        #     avg("retention_rate").alias("avg_retention_rate")
        # ).show()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()