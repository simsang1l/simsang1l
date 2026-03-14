import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min, max, percentile_approx, count, sum as spark_sum, countDistinct
import logging
from dotenv import load_dotenv
import os

# 1. 환경 변수에서 PostgreSQL 자격 증명 가져오기
load_dotenv()
POSTGRES_HOST = os.getenv('host')
POSTGRES_PORT = os.getenv('port')
POSTGRES_DB = os.getenv('dbname')
POSTGRES_USER = os.getenv('user')
POSTGRES_PASSWORD = os.getenv('password')
POSTGRES_TABLE = os.getenv('POSTGRES_TABLE', 'your_schema.dicom_cdm')



# 2. 로깅 설정
os.makedirs('./log', exist_ok=True)
logging.basicConfig(
    filename='./log/parquet_postgres_join.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# 결과 디렉토리 생성
os.makedirs('./result', exist_ok=True)


try:
    # 3. SparkSession 생성 - 로컬 머신의 사양에 따라 메모리 설정을 조정
    spark = SparkSession.builder\
        .appName("ParquetGroupStatistics")\
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "64g") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4")\
        .config("spark.sql.shuffle.partitions", "2000")\
        .config("spark.memory.fraction", "0.8")\
        .config("spark.memory.storageFraction", "0.3")\
        .config("spark.sql.shuffle.compression.codec", "snappy")\
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true")\
        .config("spark.shuffle.file.buffer", "1m")\
        .config("spark.local.dir", "/tmp/spark_tmp")\
        .getOrCreate()
    
    
    logging.info("SparkSession started.")

    # Parquet 파일들이 있는 디렉토리 경로 지정 (와일드카드 사용)
    directory_path = "./output/tag_info/*.parquet"

    # 4. Parquet 파일 로드, 모든 Parquet 파일을 한 번에 로드
    df = spark.read.parquet(directory_path)
    
    # 필요한 컬럼만 선택하여 로드
    df = df.select("patient_no", "source_modality", "source_filepath", "tag_sequence", "tag_name", "tag_value")
    # series_df = df.filter(df.tag_name == 'Series Instance UID')\
    #               .select('patient_no', 'source_filepath', df.tag_value.alias('series_id'), "tag_sequence", "tag_name")
    # series_df.show(5)
    
    # key_series_count = series_df.groupBy('source_filepath').agg(countDistinct('series_id').alias('series_id_count'))
    # multiple_series_keys = key_series_count.filter(key_series_count.series_id_count > 1)
    # print(f"series_df count: {series_df.count()}, multiple_series_keys count: {multiple_series_keys.count()}")
    # if multiple_series_keys.count() > 0:
        # logging.error("Some files have multiple series_ids.")
        # raise Exception("Some files have multiple series_ids. Please check the data consistency.")


    # 5. cohort CSV 파일 읽기 - header 있고, 모든 컬럼을 문자열로 읽음
    cohort = spark.read.option("header", "true")\
                       .option("inferSchema", "false")\
                       .csv('./curation_cohort.csv')
    
    # 6. PostgreSQL 테이블 로드 with 필터 조건
    try:
        jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        connection_properties = {
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "driver": "org.postgresql.Driver"
        }

        filtered_table_query = f"(SELECT id, patient_no, seriesinstanceuid, rename_filepath, source_filepath FROM {POSTGRES_TABLE} WHERE rename_filepath is not null) AS filtered_table"
        
        # 병렬 읽기 설정
        partition_column = "id"  # 테이블의 분할에 사용할 컬럼
        lower_bound = 1          # 해당 컬럼의 최소값
        upper_bound = 1000000    # 해당 컬럼의 최대값
        num_partitions = 20      # 읽을 파티션 수

        postgres_df = spark.read.jdbc(
            url=jdbc_url,
            table=filtered_table_query,
            column=partition_column,
            lowerBound=lower_bound,
            upperBound=upper_bound,
            numPartitions=num_partitions,
            properties=connection_properties
        )
        
        postgres_df = postgres_df.select(
            [col(c).alias(f"postgres_{c}") if c in df.columns else col(c) for c in postgres_df.columns]
        )
        
        total_postgres_rows = postgres_df.count()
        logging.info(f"PostgreSQL data loaded from {POSTGRES_TABLE}. Total rows: {total_postgres_rows}")
        print(f"PostgreSQL data loaded. Total rows: {total_postgres_rows}")
    except Exception as e:
        logging.error(f"Failed to load PostgreSQL data: {e}")
        raise
    
    # 데이터 타입 확인 및 필요한 경우 캐스팅
    df = df.withColumn("patient_no", col("patient_no").cast("string"))
    # series_df = series_df.withColumn("patient_no", col("patient_no").cast("string"))
    postgres_df = postgres_df.withColumn("postgres_patient_no", col("postgres_patient_no").cast("string"))
    cohort = cohort.withColumn("person_source_value", col("person_source_value").cast("string"))
    
    # 데이터 확인
    logging.info(f"df 컬럼 목록: {df.columns}")
    logging.info(f"cohort 컬럼 목록: {cohort.columns}")
    total_rows_df = df.count()
    total_rows_cohort = cohort.count()
    logging.info(f"df 전체 행 수: {total_rows_df}")
    logging.info(f"cohort 전체 행 수: {total_rows_cohort}")
    print("df 컬럼 목록:", df.columns)
    print("cohort 컬럼 목록:", cohort.columns)
    print("df 전체 행 수:", total_rows_df)
    print("cohort 전체 행 수:", total_rows_cohort)
    # logging.info(f"series_df 컬럼 목록: {series_df.columns}, {series_df.count()}")
    logging.info(f"postgres_df 컬럼 목록: {postgres_df.columns}")

    # 6. 그룹화 기준 설정
    group_columns = ["category", "source_modality", "seriesinstanceuid"]

    
    # 7. 조인 수행
    # df와 cohort를 키 컬럼을 기준으로 inner join
    # joined_df = series_df.join(cohort, cohort.person_source_value == series_df.patient_no, how="inner")
    # logging.info(f'join된 데이터 건수 joined_df:{joined_df.count()}')
    joined_df = postgres_df.join(
            cohort,
            cohort.person_source_value == postgres_df.postgres_patient_no,
            how="inner"
        )
    df_with_series = joined_df.join(df, joined_df.postgres_source_filepath == df.source_filepath, how = "inner")
    logging.info(f'join된 데이터 건수 joined_df:{joined_df.count()}')
    logging.info(f'df series columns:{df_with_series.columns}')
    
    logging.info("df와 cohort를 person_source_value와 patient_no로 조인 완료.")

    # 8. 그룹화 및 행 개수 계산
    # 각 카테고리 내에서 각 key의 행 개수를 계산
    group_key_counts_df = df_with_series.groupBy(*group_columns).agg(count("tag_name").alias("tag_count"))
    logging.info("그룹별 키 행 개수 계산 완료.")
    
    # 결과 확인 (선택 사항)
    group_key_counts_df.show(5)

    
    # 카테고리별로 통계 계산
    stats_df = group_key_counts_df.groupBy(group_columns).agg(
        min("tag_count").alias("min_tag_count"),
        max("tag_count").alias("max_tag_count"),
        mean("tag_count").alias("mean_tag_count"),
        stddev("tag_count").alias("stddev_tag_count"),
        percentile_approx("tag_count", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 10000).alias("percentiles"),
        count("seriesinstanceuid").alias('total_series')
    )
    logging.info("카테고리별 통계 계산 완료.")

    # 통계 결과 수집
    stats = stats_df.collect()[0]
    percentiles = stats['percentiles']
    
    print("시리즈별 태그 수 통계 정보:")
    print(f"전체 시리즈 수: {stats['total_series']}")
    print(f"최소 태그 수: {stats['min_tag_count']}")
    print(f"최대 태그 수: {stats['max_tag_count']}")
    print(f"평균 태그 수: {stats['mean_tag_count']}")
    print(f"표준편차: {stats['stddev_tag_count']}")
    for i, p in enumerate(range(10, 101, 10)):
        print(f"{p}% 백분위수: {percentiles[i]}")
    
    logging.info(f"Total series: {stats['total_series']}, Min tag count: {stats['min_tag_count']}, "
                 f"Max tag count: {stats['max_tag_count']}, Mean: {stats['mean_tag_count']}, "
                 f"StdDev: {stats['stddev_tag_count']}, Percentiles: {percentiles}")
    
    # 12. 통계 정보를 CSV 파일로 저장
    from pyspark.sql.functions import lit
    
    # 백분위수 컬럼 분리
    for i, p in enumerate(range(10, 101, 10)):
        stats_df = stats_df.withColumn(f"percentile_{p}", col("percentiles")[i])
    stats_df = stats_df.drop("percentiles")
    
    # CSV로 저장
    output_csv = "./result/tag_counts_statistics.csv"
    stats_df.coalesce(1).write.csv(output_csv, header=True, mode="overwrite")
    logging.info(f"통계 정보 CSV 파일로 저장 완료: {output_csv}")

except Exception as e:
    logging.error(f"Error during processing: {e}")
    print(f"Error: {e}")
finally:
    # SparkSession 종료
    spark.stop()
    logging.info("SparkSession 종료됨.")