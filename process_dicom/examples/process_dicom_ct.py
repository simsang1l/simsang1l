import os, argparse, io
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pydicom
import dicom2nifti

import psycopg2
from psycopg2 import sql, pool as pgpool

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dicom_key import Base, create_dicom_key_class, Dicomtags, create_dicom_key_class_for_ct
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
from glob import glob

import asyncio
import aiofiles

from multiprocessing import Pool, cpu_count

import asyncpg

from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq


def postgres_connection():
    """
    데이터베이스에 연결하고 연결 객체를 반환합니다.

    :param dbname: 데이터베이스 이름
    :param user: 사용자 이름
    :param password: 비밀번호
    :param host: 데이터베이스 호스트
    :param port: 데이터베이스 포트
    :return: psycopg2 연결 객체
    """
    load_dotenv()

    # connect postgresql
    dbname = os.environ.get("dbname")
    user = os.environ.get("user")
    password = os.environ.get("password")
    host = os.environ.get("host")
    port = os.environ.get("port")
    conn = psycopg2.connect(dbname = dbname,
                            user = user,
                            password = password,
                            host = host,
                            port = port)
    
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        return None

def process_dicom_file(firstfile, patient_no, workdate, execdate, exectime, modality, tags, model):
    '''
    dicom file을 읽어서 pixel데이터는 제외하고 태그를 추출하는 함수
    '''
    # firstfile, patient_no, execdate, exectime, modality, tags = file_info
    extracted_tags = {}
    img = pydicom.dcmread(firstfile, force=True, stop_before_pixels=True)
    extracted_tags["patient_no"] = patient_no
    extracted_tags["workdate"] = workdate
    extracted_tags["execdate"] = execdate
    extracted_tags["exectime"] = exectime
    extracted_tags["file_size"] = os.path.getsize(firstfile)
    extracted_tags["source_filepath"] = firstfile
    extracted_tags["source_filename"] = os.path.splitext(os.path.basename(firstfile))[0]
    extracted_tags["source_file_extension"] = os.path.splitext(firstfile)[1][1:]
    extracted_tags["source_modality"] = modality

    for tag in tags:      
        if tag in img:
            extracted_tags[tag.lower()] = img[tag].value
        else:
            extracted_tags[tag.lower()] = "Tag없음"
    
    StudyDescription_tag = "StudyDescription"
    StudyDescription_column = "studydescription_embedding"    
    if StudyDescription_tag  in img :
        extracted_tags[StudyDescription_column] = None # model.encode(img[StudyDescription_tag].value).tolist()
    else :     
        extracted_tags[StudyDescription_column] = None

    RequestedProcedureDescription_tag = "RequestedProcedureDescription"
    RequestedProcedureDescription_column = "requestedproceduredescription_embedding"
    if RequestedProcedureDescription_tag in img:
        extracted_tags[RequestedProcedureDescription_column] = None #model.encode(img[RequestedProcedureDescription_tag].value).tolist()
    else :     
        extracted_tags[RequestedProcedureDescription_column] = None
    
    extracted_tags["rename_filepath"] = None
    extracted_tags["person_id"] = None
    extracted_tags["orddate"] = None
    extracted_tags["ordseqno"] = None

    return extracted_tags

def load_dicom_tags(start_path, tags, start_date: int, end_date: int, model=None):
    print('load_dicom_tags start...')
    filelist = []
    modal = ["CR", "MR", "CT"]
    folders = sorted([folder for folder in os.listdir(start_path) 
                      if datetime.strptime(folder[:8], '%Y%m%d') >= datetime.strptime(str(start_date), '%Y%m%d')
                        and datetime.strptime(folder[:8], '%Y%m%d') <= datetime.strptime(str(end_date), '%Y%m%d')])
    
    for folder in tqdm(folders, mininterval=60.0):
        if folder:
            work_path = os.path.join(start_path, folder)
            workdate = folder.split('_')[0]

            if start_date <= int(workdate) <= end_date:
                for dir_1 in os.listdir(work_path):
                    if dir_1:
                        data_path = os.path.join(work_path, dir_1)
                        patient_no, execdate, exectime, modality = dir_1.split('_')

                        if modality in modal:
                            data_list = os.listdir(data_path)

                            for d in data_list:
                                filename = d
                                firstfile = os.path.join(data_path, filename)

                                if os.path.isfile(firstfile):
                                    extracted_tags = process_dicom_file(firstfile, patient_no, workdate, execdate, exectime, modality, tags, model)
                                    filelist.append(extracted_tags)
            elif int(workdate) >= end_date:
                break

    dicom_tags = pd.DataFrame(filelist)
    for tag in tags:
        column = tag.lower()
        dicom_tags[column] = dicom_tags[column].apply(str)
    
    # dicom_tags.to_csv("load_dicom_tags_result.csv", index = False)
    print('dicom_tags shape in load_dicom_tags;;',dicom_tags.shape)
    return dicom_tags

# DICOM 태그 필터링 및 비동기 I/O 최적화
async def process_dicom_file_async(firstfile, patient_no, workdate, execdate, exectime, modality, tags):
    extracted_tags = {}
    async with aiofiles.open(firstfile, 'rb') as f:
        content = await f.read()  # 비동기 파일 읽기
        
    # 바이트 데이터를 파일 객체처럼 처리
    dicom_file = io.BytesIO(content)
    
    # pydicom.dcmread에 파일 객체 전달
    img = pydicom.dcmread(dicom_file, force=True, stop_before_pixels=True)  # 이미지 데이터 제외
    
    extracted_tags["patient_no"] = patient_no
    extracted_tags["workdate"] = workdate
    extracted_tags["execdate"] = execdate
    extracted_tags["exectime"] = exectime
    extracted_tags["file_size"] = os.path.getsize(firstfile)
    extracted_tags["source_filepath"] = firstfile
    extracted_tags["source_filename"] = os.path.splitext(os.path.basename(firstfile))[0]
    extracted_tags["source_file_extension"] = os.path.splitext(firstfile)[1][1:]
    extracted_tags["source_modality"] = modality

    for tag in tags:      
        extracted_tags[tag.lower()] = img.get(tag, "Tag없음")
    
    # StudyDescription 처리
    StudyDescription_tag = "StudyDescription"
    StudyDescription_column = "studydescription_embedding"    
    if StudyDescription_tag  in img :
        extracted_tags[StudyDescription_column] = None # model.encode(img[StudyDescription_tag].value).tolist()
    else :     
        extracted_tags[StudyDescription_column] = None

    # RequestedProcedureDescription 처리
    RequestedProcedureDescription_tag = "RequestedProcedureDescription"
    RequestedProcedureDescription_column = "requestedproceduredescription_embedding"
    if RequestedProcedureDescription_tag in img:
        extracted_tags[RequestedProcedureDescription_column] = None #model.encode(img[RequestedProcedureDescription_tag].value).tolist()
    else :     
        extracted_tags[RequestedProcedureDescription_column] = None
    
    extracted_tags["rename_filepath"] = None
    extracted_tags["person_id"] = None
    extracted_tags["orddate"] = None
    extracted_tags["ordseqno"] = None

    return extracted_tags

async def load_dicom_tags_async(start_path, tags, start_date: int, end_date: int):
    print('load_dicom_tags_async start...')
    filelist = []
    modal = ["CR", "MR", "CT"]
    # 날짜 범위에 맞는 폴더만 선택
    folders = sorted([folder for folder in os.listdir(start_path)
                      if datetime.strptime(folder[:8], '%Y%m%d') >= datetime.strptime(str(start_date), '%Y%m%d')
                      and datetime.strptime(folder[:8], '%Y%m%d') <= datetime.strptime(str(end_date), '%Y%m%d')])

    tasks = []
    for folder in tqdm(folders, mininterval=60.0):
        if folder:
            work_path = os.path.join(start_path, folder)
            workdate = folder.split('_')[0]

            if start_date <= int(workdate) <= end_date:
                for dir_1 in os.listdir(work_path):
                    if dir_1:
                        data_path = os.path.join(work_path, dir_1)
                        patient_no, execdate, exectime, modality = dir_1.split('_')

                        if modality in modal:
                            data_list = os.listdir(data_path)

                            for filename in data_list:
                                firstfile = os.path.join(data_path, filename)

                                if os.path.isfile(firstfile):
                                    tasks.append(process_dicom_file_async(firstfile, patient_no, workdate, execdate, exectime, modality, tags))
            elif int(workdate) >= end_date:
                break

    # 비동기적으로 모든 파일 처리
    filelist = await asyncio.gather(*tasks)
    
    # 처리된 결과를 DataFrame으로 변환
    dicom_tags = pd.DataFrame(filelist)
    
    for tag in tags:
        column = tag.lower()
        dicom_tags[column] = dicom_tags[column].apply(str)
    
    # dicom_tags.to_csv("load_dicom_tags_result.csv", index = False)
    print('dicom_tags shape in load_dicom_tags;;',dicom_tags.shape)
    return dicom_tags


def process_dicom_file_multiprocessing(file_info):
    """
    dicom tag들을 dictionary로 저장하는 형태
    """
    firstfile, patient_no, workdate, execdate, exectime, modality, tags = file_info
    extracted_tags = {}
    tag_info = []
    
    img = pydicom.dcmread(firstfile, force=True, stop_before_pixels=True)
    extracted_tags["patient_no"] = patient_no
    extracted_tags["workdate"] = workdate
    extracted_tags["execdate"] = execdate
    extracted_tags["exectime"] = exectime
    extracted_tags["file_size"] = str(os.path.getsize(firstfile))
    extracted_tags["source_filepath"] = firstfile
    extracted_tags["source_filename"] = os.path.splitext(os.path.basename(firstfile))[0]
    extracted_tags["source_file_extension"] = os.path.splitext(firstfile)[1][1:]
    extracted_tags["source_modality"] = modality

    # 태그의 시퀀스(태그 번호), 이름, 값 추출
    tags_list = []
    
    for elem in img.file_meta.iterall():
        tag_sequence = f'({elem.tag.group:04X},{elem.tag.element:04X})'
        tag_name = elem.name
        tag_value = str(elem.value)
        if isinstance(tag_value, pydicom.multival.MultiValue):
            tag_value = list(tag_value)
        
        tags_list.append({"tag_sequence": tag_sequence,
                           "tag_name": tag_name,
                           "tag_value": tag_value })
        
        
    for elem in img.iterall():
        tag_sequence = f'({elem.tag.group:04X},{elem.tag.element:04X})'
        tag_name = elem.name
        tag_value = str(elem.value)
        if isinstance(tag_value, pydicom.multival.MultiValue):
            tag_value = list(tag_value)
        
        tags_list.append({ "tag_sequence": tag_sequence,
                           "tag_name": tag_name,
                           "tag_value": tag_value })
        
    tag_info.append({
        "patient_no": patient_no ,
        "workdate": workdate,
        "execdate": execdate ,
        "exectime": exectime,
        "file_size": str(os.path.getsize(firstfile)) ,
        "source_filepath": firstfile ,
        "source_filename": os.path.splitext(os.path.basename(firstfile))[0] ,
        "source_file_extension": os.path.splitext(firstfile)[1][1:],
        "source_modality": modality ,
        "tags": tags_list
    })
    
    for tag in tags:
        tag_lower = tag.lower()
        extracted_tags[tag_lower] = "Tag없음"  # 기본값 설정
        if tag in img:
            extracted_tags[tag_lower] = img[tag].value  # 값이 있으면 덮어쓰기

    
    StudyDescription_tag = "StudyDescription"
    StudyDescription_column = "studydescription_embedding"    
    if StudyDescription_tag  in img :
        extracted_tags[StudyDescription_column] = None # model.encode(img[StudyDescription_tag].value).tolist()
    else :     
        extracted_tags[StudyDescription_column] = None

    RequestedProcedureDescription_tag = "RequestedProcedureDescription"
    RequestedProcedureDescription_column = "requestedproceduredescription_embedding"
    if RequestedProcedureDescription_tag in img:
        extracted_tags[RequestedProcedureDescription_column] = None #model.encode(img[RequestedProcedureDescription_tag].value).tolist()
    else :     
        extracted_tags[RequestedProcedureDescription_column] = None
    
    extracted_tags["rename_filepath"] = None
    extracted_tags["person_id"] = None
    extracted_tags["orddate"] = None
    extracted_tags["ordseqno"] = None

    return extracted_tags, tag_info


def process_dicom_file_each_taginfo_multiprocessing(file_info):
    """
    multiprocessing을 통해 dicom파일에서 tag들을 dictionary로 저장하는 함수
    """
    firstfile, patient_no, workdate, execdate, exectime, modality, tags = file_info
    extracted_tags = {}
    tag_info = []
    
    img = pydicom.dcmread(firstfile, force=True, stop_before_pixels=True)
    extracted_tags["patient_no"] = patient_no
    extracted_tags["workdate"] = workdate
    extracted_tags["execdate"] = execdate
    extracted_tags["exectime"] = exectime
    extracted_tags["file_size"] = str(os.path.getsize(firstfile))
    extracted_tags["source_filepath"] = firstfile
    extracted_tags["source_filename"] = os.path.splitext(os.path.basename(firstfile))[0]
    extracted_tags["source_file_extension"] = os.path.splitext(firstfile)[1][1:]
    extracted_tags["source_modality"] = modality
    
    # 공통 메타데이터를 미리 계산하여 중복 계산 방지
    file_size = str(os.path.getsize(firstfile))
    source_filename = os.path.splitext(os.path.basename(firstfile))[0]
    source_file_extension = os.path.splitext(firstfile)[1][1:]
    studyinstanceuid = img['StudyInstanceUID'].value if 'StudyInstanceUID' in img else 'Tag없음'
    seriesinstanceuid = img['SeriesInstanceUID'].value if 'SeriesInstanceUID' in img else 'Tag없음'
    
    # 공통 메타데이터 딕셔너리 생성
    common_metadata = {
        "patient_no": patient_no,
        "workdate": workdate,
        "execdate": execdate,
        "exectime": exectime,
        "file_size": file_size,
        "source_filepath": firstfile,
        "source_filename": source_filename,
        "source_file_extension": source_file_extension,
        "source_modality": modality,
        "studyinstanceuid": studyinstanceuid,
        "seriesinstanceuid": seriesinstanceuid
    }
    
    # 태그 처리 헬퍼 함수
    def process_tag_element(elem):
        """태그 요소를 처리하여 tag_info 딕셔너리 생성"""
        tag_sequence = f'({elem.tag.group:04X},{elem.tag.element:04X})'
        tag_name = elem.name
        tag_value = str(elem.value)
        if isinstance(tag_value, pydicom.multival.MultiValue):
            tag_value = list(tag_value)
        
        tag_dict = common_metadata.copy()
        tag_dict.update({
            "tag_sequence": tag_sequence,
            "tag_name": tag_name,
            "tag_value": tag_value
        })
        return tag_dict
    
    # 파일 메타데이터 태그 처리
    for elem in img.file_meta.iterall():
        tag_info.append(process_tag_element(elem))
        
    # 일반 데이터셋 태그 처리
    for elem in img.iterall():
        tag_info.append(process_tag_element(elem))
    
    for tag in tags:
        tag_lower = tag.lower()
        extracted_tags[tag_lower] = "Tag없음"  # 기본값 설정
        if tag in img:
            extracted_tags[tag_lower] = img[tag].value  # 값이 있으면 덮어쓰기

    # 임베딩 필드 초기화 (현재는 주석 처리되어 항상 None)
    extracted_tags["studydescription_embedding"] = None  # model.encode(img[StudyDescription_tag].value).tolist()
    extracted_tags["requestedproceduredescription_embedding"] = None  # model.encode(img[RequestedProcedureDescription_tag].value).tolist()
    
    # 추가 필드 초기화
    extracted_tags["rename_filepath"] = None
    extracted_tags["person_id"] = None
    extracted_tags["orddate"] = None
    extracted_tags["ordseqno"] = None

    return extracted_tags, tag_info

# 멀티프로세싱으로 여러 DICOM 파일 처리
def load_dicom_tags_multiprocessing(start_path, tags, start_date: int, end_date: int, modal = ["CR", "MR", "CT"]):
    '''
    원본의 경로 및 파일명 정보, dicom tag정보를 추출하여 데이터프레임으로 변환하는 함수
    '''
    print('load_dicom_tags start...')    
    
    # 날짜 범위에 맞는 폴더 선택
    folders = sorted([folder for folder in os.listdir(start_path)
                      if folder[:8].isdigit() and start_date <= int(folder[:8]) <= end_date])
    
    # 폴더 정보 로깅
    logging.info(f"Found {len(folders)} folders for date range {start_date}-{end_date}")
    logging.info(f"Folders: {folders}")
    for folder in folders:
        folder_path = os.path.join(start_path, folder)
        ct_folders = [d for d in os.listdir(folder_path) if d.endswith('_CT')]
        logging.info(f"Folder {folder}: {len(ct_folders)} CT subfolders")
        for ct_folder in ct_folders:
            ct_path = os.path.join(folder_path, ct_folder)
            dcm_count = len([f for f in os.listdir(ct_path) if f.endswith('.dcm')])
            logging.info(f"  - {ct_folder}: {dcm_count} DICOM files")
    
    # 파일 정보를 담을 리스트 생성
    files_to_process = []
    
    # 모든 DICOM 파일의 경로 및 정보 수집
    for folder in tqdm(folders, mininterval=60.0):
        work_path = os.path.join(start_path, folder)
        workdate = folder.split('_')[0]

        for dir_1 in os.listdir(work_path):
            if dir_1:
                data_path = os.path.join(work_path, dir_1)
                patient_no, execdate, exectime, modality = dir_1.split('_')

                if modality in modal:
                    data_list = os.listdir(data_path)

                    for filename in data_list:
                        firstfile = os.path.join(data_path, filename)

                        if os.path.isfile(firstfile):
                            # 처리할 파일 정보를 튜플로 담아서 리스트에 추가
                            files_to_process.append((firstfile, patient_no, workdate, execdate, exectime, modality, tags))

    # 멀티프로세싱을 통해 파일 병렬 처리
    num_processes = cpu_count()  # 시스템의 CPU 코어 수에 따라 프로세스 수 설정
    print(f"Using {num_processes} processes for parallel processing")

    # Pool을 사용하여 병렬로 파일 처리
    with Pool(processes=num_processes) as pool:
        # results = list(tqdm(pool.imap(process_dicom_file_multiprocessing, files_to_process), total=len(files_to_process)))
        results = list(tqdm(pool.imap(process_dicom_file_each_taginfo_multiprocessing, files_to_process), total=len(files_to_process), mininterval = 60.0))

    # 유효한 결과만 수집
    basic_info_list = []
    tag_info_list = []
    for basic_info, tag_info in results:
        if basic_info is not None:
            basic_info_list.append(basic_info)
        if tag_info is not None:
            tag_info_list.extend(tag_info)
    
    # DataFrame으로 변환
    basic_info_df = pd.DataFrame(basic_info_list)
    tag_info_df = pd.DataFrame(tag_info_list)

    # 태그 값을 문자열로 변환 (필요에 따라 유지)
    for tag in tags:
        column = tag.lower()
        if column in basic_info_df.columns:
            basic_info_df[column] = basic_info_df[column].apply(str)
        else:
            basic_info_df[column] = "Tag없음"
            
        if column in tag_info_df.columns:
            tag_info_df[column] = tag_info_df[column].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

    print('dicom_tags shape in load_dicom_tags_multiprocessing:', basic_info_df.shape, tag_info_df.shape)
    
    # basic_info_df 결과 로깅
    logging.info(f"Processing completed. Basic info shape: {basic_info_df.shape}")
    logging.info(f"Basic info columns: {list(basic_info_df.columns)}")
    logging.info(f"Basic info sample data:")
    logging.info(f"{basic_info_df.head().to_string()}")
    
    # 통계 정보 로깅
    if not basic_info_df.empty:
        logging.info(f"Total files processed: {len(basic_info_df)}")
        logging.info(f"Unique patients: {basic_info_df['patient_no'].nunique() if 'patient_no' in basic_info_df.columns else 'N/A'}")
        logging.info(f"Unique modalities: {basic_info_df['source_modality'].value_counts().to_dict() if 'source_modality' in basic_info_df.columns else 'N/A'}")
        
        # 파일 크기 통계
        if 'file_size' in basic_info_df.columns:
            file_sizes = pd.to_numeric(basic_info_df['file_size'], errors='coerce')
            logging.info(f"File size stats - Min: {file_sizes.min()}, Max: {file_sizes.max()}, Mean: {file_sizes.mean():.2f}")
    
    logging.info(f"Log file saved: {log_filename}")
    return basic_info_df, tag_info_df



def save_to_parquet(tag_df, output_dir, start_date, end_date):
    """
    태그 정보 DataFrame을 Parquet 파일로 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 태그 정보 저장 (대규모 데이터)
    tag_parquet_path = os.path.join(output_dir, f'tag_info_{start_date}_{end_date}.parquet')
    tag_df.to_parquet(tag_parquet_path, index=False, compression='snappy')
    print(f"Tag information saved to {tag_parquet_path}")


def save_tag_info_to_parquet(tag_info_df, output_dir, date):
    """
    태그 정보 DataFrame을 Parquet 파일로 저장하는 함수 (병렬 처리)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 지정
    parquet_file = os.path.join(output_dir, f'tag_info_{date}.parquet')
    
    # PyArrow Table로 변환
    table = pa.Table.from_pandas(tag_info_df)
    
    # 멀티스레딩을 이용하여 Parquet 파일 저장 (PyArrow는 내부적으로 멀티스레드를 사용)
    pq.write_table(table, parquet_file, compression='snappy')
    print(f"Tag information saved to {parquet_file}")
    
    
    # # CPU 코어 수에 맞게 데이터를 분할
    # num_processes = cpu_count()
    # df_chunks = np.array_split(tag_info_df, num_processes)
    
    # def save_chunk(df_chunk, part):
    #     """
    #     DataFrame 청크를 Parquet 파일로 저장
    #     """
    #     parquet_file = os.path.join(output_dir, f'tag_info_part_{part + 1}.parquet')
    #     df_chunk.to_parquet(parquet_file, index=False, compression='snappy')
    #     print(f"Saved {parquet_file} with {df_chunk.shape[0]} records")
    
    # # 멀티프로세싱 Pool을 사용하여 병렬로 Parquet 저장
    # with Pool(processes=num_processes) as pool:
    #     pool.starmap(save_chunk, [(df_chunks[i], i) for i in range(len(df_chunks))])    
    
    
def query_database_batch(engine, df: pd.DataFrame, key_column: str, schema: str, table_name: str) -> pd.DataFrame:
    """
    PostgreSQL 데이터베이스에서 키를 기준으로 조회하는 함수

    Parameters:
        engine: SQLAlchemy 엔진 객체
        df (pd.DataFrame): 데이터프레임
        key_column (str): 키 컬럼 이름

    Returns:
        pd.DataFrame: 데이터베이스에 없는 키 값을 가진 데이터프레임
    """
    keys = df[key_column].tolist()
    existing_keys = set()
    
    query = text(f"""
        SELECT {key_column}
        FROM {schema}.{table_name}
        WHERE {key_column} = ANY(:keys)
    """)
    with engine.connect() as connection:
        try:
            result = connection.execute(query, {'keys': keys})
            existing_keys.update(row[0] for row in result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()  # 빈 데이터프레임 반환
    
    new_data = df[~df[key_column].isin(existing_keys)]
    return new_data    
                  
def save_dicom_tags_to_postgres(input_df, batch_size: int = 100000, dicom_key_table = "dicom_metadata_ct"):
    '''
    추출한 dicom tag들을 postgreSQL에 저장하는 함수
    '''
    print('save_dicom_tags_to_postgres start...')
    load_dotenv()

    # connect postgresql
    dbname = os.environ.get("dbname")
    user = os.environ.get("user")
    password = os.environ.get("password")
    host = os.environ.get("host")
    port = os.environ.get("port")
    
    # define schema, table
    schema_name = "your_schema"
    key_column = "source_filepath"
    
    try :
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
        inspector = inspect(engine)
        table_exists = inspector.has_table(dicom_key_table, schema=schema_name)
        
        if not table_exists:
            print(table_exists)
            try:
                table_class = create_dicom_key_class_for_ct(dicom_key_table, primary_key_id=True, primary_key_source_filepath=False, autoincrement=True)
                # table_class = create_dicom_key_class(dicom_key_table, primary_key=True, autoincrement=True)
                Base.metadata.create_all(engine, tables = [table_class.__table__])
                print(f"Table '{dicom_key_table}' does not exist.")
            except SQLAlchemyError as e:
                print(f"Table '{dicom_key_table}' already exists.: {e}")
        else :
            print(f"Table '{dicom_key_table}' already exists.")
        
        with engine.begin() as connection :
            # 시퀀스 재설정
            sequence_reset_query = text(f"""
                SELECT
                    setval(pg_get_serial_sequence('{schema_name}.{dicom_key_table}', 'id'), 
                    COALESCE(MAX(id), 1))
                FROM {schema_name}.{dicom_key_table};
            """)         
            connection.execute(sequence_reset_query)
            
            for start in tqdm(range(0, len(input_df), batch_size), mininterval=60.0) :
                batch_df = input_df.iloc[start:start + batch_size]
                # 기존 데이터베이스에 없는 데이터만 필터링
                # new_data = query_database_batch(engine, batch_df, key_column, schema_name, dicom_key_table)
                
                # 새로운 데이터만 삽입
                if not batch_df.empty: # not new_data.empty:
                    
                    # new_data.to_sql(dicom_key_table, con=connection, schema=schema_name, if_exists='append', index=False)
                    batch_df.to_sql(dicom_key_table, con=connection, schema=schema_name, if_exists='append', index=False)
                else:
                    print(f"No new data to insert for batch starting at index {start}.")
                
    except SQLAlchemyError as e :
        print(f"Failed to save dataframe to PostgreSQL: {e}")



# 사용자 정의 벡터 인코더/디코더 등록
async def setup_custom_encoders(conn):
    vector_oid = await conn.fetchval("SELECT oid FROM pg_type WHERE typname = 'vector';")
    
    if not vector_oid:
        raise ValueError("Vector type not found in PostgreSQL")
    
    await conn.set_type_codec(
        'vector',  # PostgreSQL의 vector 타입
        encoder=lambda v: f'[{", ".join(map(str, v))}]' if v else None,
        decoder=lambda v: list(map(float, v[1:-1].split(','))) if v else None,
        schema='public',
        format = 'text'
    )

  
# 비동기적으로 PostgreSQL에 연결하여 대량 삽입을 처리하는 함수
async def save_batch_to_postgres_async(connection, new_data: pd.DataFrame, schema: str, table_name: str):
    if not new_data.empty:
        try:        
            # 트랜잭션 내에서 데이터 삽입
            async with connection.transaction():
                await connection.copy_records_to_table(
                    table_name,
                    records=new_data.itertuples(index=False, name=None),
                    columns=list(new_data.columns),
                    schema_name=schema
                )
        except Exception as e:
            print(f"Error during insertion: {e}")
    else:
        print("No new data to insert.")
        
# 비동기적으로 PostgreSQL에 연결하여 키 조회 및 중복 확인
async def query_existing_keys_async(connection, df: pd.DataFrame, key_column: str, schema: str, table_name: str, first_batch: bool):
    keys = df[key_column].tolist()
    existing_keys = set()

    try:
        async with connection.transaction():
            # 첫 번째 배치에서만 임시 테이블을 생성
            if first_batch:
                await connection.execute("""
                    DROP TABLE IF EXISTS temp_keys;  -- 이전에 임시 테이블이 남아 있으면 삭제
                    CREATE TEMPORARY TABLE temp_keys (key TEXT);
                """)
                
            # 임시 테이블을 매번 비우고 새로운 키 삽입
            await connection.executemany("""
                DELETE FROM temp_keys;
            """, [])
            
            await connection.executemany("""
                INSERT INTO temp_keys (key) VALUES ($1);
            """, [(k,) for k in keys])

            # 조인을 통해 중복된 키 조회
            result = await connection.fetch(f"""
                SELECT {key_column}
                FROM {schema}.{table_name} db_keys
                INNER JOIN temp_keys tk ON db_keys.{key_column} = tk.key;
            """)
            existing_keys.update([row[0] for row in result])

    except Exception as e:
        print(f"Error querying existing keys: {e}")
    
    new_data = df[~df[key_column].isin(existing_keys)]
    return new_data


# 테이블이 존재하는지 확인하고 없으면 생성하는 함수
def create_table_if_not_exists(engine, schema_name, table_name):
    try:
        inspector = inspect(engine)
        table_exists = inspector.has_table(table_name, schema=schema_name)
        if not table_exists:
            print(f"Table '{table_name}' does not exist. Creating it now...")
            
            # 테이블 생성 로직
            # with engine.begin() as connection:
            table_class = create_dicom_key_class(table_name, primary_key=True, autoincrement=True)
            Base.metadata.create_all(engine, tables = [table_class.__table__])
            print(f"Table '{table_name}' created successfully.")
        else:
            print(f"Table '{table_name}' already exists.")
    except SQLAlchemyError as e:
        print(f"Failed to check or create table: {e}")

def create_table_if_not_exists_dicomtags(engine, table_class):
    try:
        inspector = inspect(engine)
        schema_name = table_class.__table_args__.get('schema', None)
        table_name = table_class.__tablename__
        table_exists = inspector.has_table(table_name, schema=schema_name)
        if not table_exists:
            print(f"Table '{schema_name}.{table_name}' does not exist. Creating it now...")
            # 테이블 생성
            table_class.__table__.create(bind=engine)
            print(f"Table '{schema_name}.{table_name}' created successfully.")
        else:
            print(f"Table '{schema_name}.{table_name}' already exists.")
    except SQLAlchemyError as e:
        print(f"Failed to check or create table: {e}")
                
        
# 비동기적으로 대량 데이터 저장을 처리
async def save_dicom_key_source_to_postgres_async(input_df, batch_size: int = 10000):
    load_dotenv()

    # PostgreSQL 연결 정보
    dbname = os.environ.get("dbname")
    user = os.environ.get("user")
    password = os.environ.get("password")
    host = os.environ.get("host")
    port = os.environ.get("port")

    schema_name = "your_schema"
    dicom_key_table = "dicom_metadata"
    key_column = "source_filepath"

    # SQLAlchemy 엔진을 통해 테이블 존재 여부 확인 및 테이블 생성
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    create_table_if_not_exists(engine, schema_name, dicom_key_table)
    
    # 비동기 PostgreSQL 연결 풀 생성 (최대 연결 수 제한)
    pool = await asyncpg.create_pool(user=user, password=password, database=dbname, host=host, port=port, max_size=10)

    try:
        async with pool.acquire() as connection:
            # await setup_custom_encoders(connection)  # 커스텀 인코더 등록
            
            # 시퀀스 재설정
            await connection.execute(f"""
                SELECT setval(pg_get_serial_sequence('{schema_name}.{dicom_key_table}', 'id'), COALESCE(MAX(id), 1))
                FROM {schema_name}.{dicom_key_table};
            """)

            # 배치로 데이터 처리
            first_batch = True
            for start in tqdm(range(0, len(input_df), batch_size), mininterval=60.0):
                batch_df = input_df.iloc[start:start + batch_size]
                
                # 중복되지 않은 데이터만 필터링
                new_data = await query_existing_keys_async(connection, batch_df, key_column, schema_name, dicom_key_table, first_batch)

                # 새로운 데이터만 삽입
                await save_batch_to_postgres_async(connection, new_data, schema_name, dicom_key_table)
                
                # 첫 배치 후, first_batch를 False로 설정하여 임시 테이블을 재사용
                first_batch = False

    except Exception as e:
        print(f"Failed to save data to PostgreSQL: {e}")

    finally:
        await pool.close()
        
        
# 비동기적으로 대량 데이터 저장을 처리
async def save_dicom_tags_to_postgres_async(input_df, batch_size: int = 10000, 
                                            schema_name = "your_schema", dicom_tag_table = "dicom_tags", key_column = "source_filepath"):
    load_dotenv()

    # PostgreSQL 연결 정보
    dbname = os.environ.get("dbname")
    user = os.environ.get("user")
    password = os.environ.get("password")
    host = os.environ.get("host")
    port = os.environ.get("port")

    # SQLAlchemy 엔진을 통해 테이블 존재 여부 확인 및 테이블 생성
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    create_table_if_not_exists_dicomtags(engine, Dicomtags)
    
    # 비동기 PostgreSQL 연결 풀 생성 (최대 연결 수 제한)
    pool = await asyncpg.create_pool(user=user, password=password, database=dbname, host=host, port=port, max_size=10)

    try:
        async with pool.acquire() as connection:
            
            # 시퀀스 재설정
            await connection.execute(f"""
                SELECT setval(pg_get_serial_sequence('{schema_name}.{dicom_tag_table}', 'id'), COALESCE(MAX(id), 1))
                FROM {schema_name}.{dicom_tag_table};
            """)

            # 배치로 데이터 처리
            first_batch = True
            for start in tqdm(range(0, len(input_df), batch_size), mininterval=60.0):
                batch_df = input_df.iloc[start:start + batch_size]
                
                # 중복되지 않은 데이터만 필터링
                new_data = await query_existing_keys_async(connection, batch_df, key_column, schema_name, dicom_tag_table, first_batch)

                # 새로운 데이터만 삽입
                await save_batch_to_postgres_async(connection, new_data, schema_name, dicom_tag_table)
                
                # 첫 배치 후, first_batch를 False로 설정하여 임시 테이블을 재사용
                first_batch = False

    except Exception as e:
        print(f"Failed to save data to PostgreSQL: {e}")

    finally:
        await pool.close()


def save_batch_to_postgres(session, new_data: pd.DataFrame, table_class):
    """
    데이터프레임을 PostgreSQL에 배치 삽입합니다.
    
    Args:
        session: SQLAlchemy 세션 객체.
        new_data (pd.DataFrame): 삽입할 데이터가 포함된 데이터프레임.
        table_class: SQLAlchemy 테이블 클래스.
    """
    if not new_data.empty:
        try:
            # 데이터프레임을 딕셔너리 목록으로 변환
            records = new_data.to_dict(orient='records')
            
            # JSONB 컬럼 확인 및 직렬화
            for record in records:
                if 'tags' in record and isinstance(record['tags'], list):
                    # JSONB 컬럼을 Python list of dict로 유지
                    pass  # 이미 올바른 형식으로 변환됨
                else:
                    # 필요한 경우, JSON 문자열을 dict로 변환
                    record['tags'] = json.loads(record['tags']) if isinstance(record['tags'], str) else record['tags']
    
            # 데이터 삽입
            session.bulk_insert_mappings(table_class, records)
        except SQLAlchemyError as e:
            print(f"Error during insertion: {e}")
            session.rollback()
    else:
        print("No new data to insert.")
        
        
def query_existing_keys(session, df: pd.DataFrame, key_column: str, schema: str, table_name: str, first_batch: bool):
    """
    데이터프레임의 키 컬럼을 기준으로 기존 데이터베이스에 존재하는 키를 조회하고,
    중복되지 않은 데이터를 반환합니다.
    
    Args:
        session: SQLAlchemy 세션 객체.
        df (pd.DataFrame): 입력 데이터프레임.
        key_column (str): 키 컬럼 이름.
        schema (str): 스키마 이름.
        table_name (str): 테이블 이름.
        first_batch (bool): 첫 번째 배치 여부.
    
    Returns:
        pd.DataFrame: 중복되지 않은 데이터프레임.
    """
    keys = df[key_column].tolist()
    existing_keys = set()

    try:
        if first_batch:
            # 첫 번째 배치에서만 임시 테이블 생성
            session.execute(text("""
                DROP TABLE IF EXISTS temp_keys;
                CREATE TEMPORARY TABLE temp_keys (key TEXT);
            """))
        
        # 임시 테이블 비우기
        session.execute(text("TRUNCATE TABLE temp_keys;"))
        
        # 새로운 키 삽입
        insert_query = text("INSERT INTO temp_keys (key) VALUES (:key)")
        data = [{'key': k} for k in keys]
        session.execute(insert_query, data)

        # 조인을 통해 중복된 키 조회
        existing_keys_query = text(f"""
            SELECT db_keys.{key_column}
            FROM {schema}.{table_name} db_keys
            INNER JOIN temp_keys tk ON db_keys.{key_column} = tk.key;
        """)
        result = session.execute(existing_keys_query)
        fetched_keys = result.fetchall()  # fetchall()을 사용하여 모든 결과를 가져옴
        existing_keys.update([row[0] for row in fetched_keys])

    except SQLAlchemyError as e:
        print(f"Error querying existing keys: {e}")
    
    # 중복되지 않은 데이터 필터링
    new_data = df[~df[key_column].isin(existing_keys)]
    return new_data


def save_dicom_tags_source(input_df, batch_size: int = 10000, 
                            schema_name = "your_schema", dicom_tag_table = "dicom_tags", key_column = "source_filepath"):
    load_dotenv()

    # PostgreSQL 연결 정보
    dbname = os.environ.get("dbname")
    user = os.environ.get("user")
    password = os.environ.get("password")
    host = os.environ.get("host")
    port = os.environ.get("port")

    # SQLAlchemy 엔진을 통해 테이블 존재 여부 확인 및 테이블 생성
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    create_table_if_not_exists_dicomtags(engine, Dicomtags)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 시퀀스 재설정 (이 부분은 트랜잭션 밖에서 실행하도록 별도 연결 사용 권장)
        with engine.connect() as conn:
            conn.execute(text(f"""
                SELECT setval(pg_get_serial_sequence('{schema_name}.{dicom_tag_table}', 'id'), COALESCE(MAX(id), 1))
                FROM {schema_name}.{dicom_tag_table};
            """))
        
        # 세션 내에서 배치로 데이터 처리
        # 트랜잭션을 명시적으로 시작
        with session.begin():
        
            first_batch = True
            for start in tqdm(range(0, len(input_df), batch_size), mininterval=60.0):
                batch_df = input_df.iloc[start:start + batch_size]
                
                # 중복되지 않은 데이터만 필터링
                new_data = query_existing_keys(session, batch_df, key_column, schema_name, dicom_tag_table, first_batch)

                # 새로운 데이터만 삽입
                save_batch_to_postgres(session, new_data, Dicomtags)
                
                # 첫 배치 후, first_batch를 False로 설정하여 임시 테이블을 재사용
                first_batch = False

            session.execute(text("DROP TABLE IF EXISTS temp_keys;"))
            
            # 모든 배치 삽입 후 커밋
            session.commit()
            print("모든 데이터 삽입 성공")
        
    except SQLAlchemyError  as e:
        print(f"Failed to save data to PostgreSQL: {e}")
        session.rollback()

    finally:
        session.close()
           
        
def format_date(date_int):
    date_str = str(date_int)
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"


def labeling_mr(row):
    scanningsequence = row["scanningsequence"]
    repetitiontime = row["repetitiontime"]
    seriesdescription = row["seriesdescription"]
    acquisitioncontrast = row["acquisitioncontrast"]
    
    repetitiontime = pd.to_numeric(repetitiontime, errors='coerce')
    if pd.isna(repetitiontime):
        repetitiontime = 0  # 또는 필요한 대체 값으로 설정
    else:
        repetitiontime = float(repetitiontime)

    # label 설정
    if (scanningsequence == 'GR' and repetitiontime <= 1000 and ("tof" not in seriesdescription.lower() and "mip" not in seriesdescription.lower() and 'reformat' not in seriesdescription.lower())) or ("t1" in seriesdescription.lower() and "ce" in seriesdescription.lower() and "bb" not in seriesdescription.lower()):
        return "t1ce"
    elif (scanningsequence == 'IR' and repetitiontime >= 2000 and repetitiontime <= 3000 and "irfse" in seriesdescription) or 'bb' in seriesdescription.lower():
        return  "bb"
    elif repetitiontime >= 4000 and "t1" not in acquisitioncontrast.lower() and "t1" not in seriesdescription.lower():
        return  "Others"
    else:
        return  "Others"



def rename_dicom(save_path, input_df, batch_size = 50000):
    """
    dicom tag와 CDM을 결합한 데이터프레임을 이용해서 dicom 파일 폴더 및 파일명 수정
    생성되는 파일
    1. 파일명 변경된 이미지
    2. 원본파일명 to 변경된 파일명 이력 (DB에 저장)

    modality가 MR인 경우
    series description tag를 이용하여 series폴더 생성 후 dcm파일 분류
    
    modality가 XR인 경우
    XR폴더에 모든 파일 저장
    
    modality가 CT인 경우..
    2024.08 현재 아직 정해지지 않아 CT폴더에 모든 파일 저장
    """
    modal = ["CT", "MR", "CR"]
    result = []
    schema_name = "your_schema"
    dicom_cdm_table = "dicom_cdm_staging"
    
    conn = postgres_connection()
    cur = conn.cursor()
    
    # Dicom과 CDM이 연결된 테이블 만들기
    # create_dicom_cdm()
    
    # 컬럼 이름 가져오기 (마지막 컬럼을 제외한 나머지)
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{dicom_cdm_table}' and table_schema = '{schema_name}'")
    columns = [col[0] for col in cur.fetchall()]

    for index, value in tqdm(input_df.iterrows(), mininterval=60.0):
        new_filepath = None
        label = None
        source_filepath = value["source_filepath"]
        
        if pd.notna(value["person_id"]) and pd.notna(value["seriesdescription"]):    
            img = pydicom.dcmread(source_filepath, stop_before_pixels=True)
            seq = source_filepath.split('.dcm')[0]
            cdm_key = f'{int(value["person_id"])}_{value["orddate"]}_{str(int(value["ordseqno"])).zfill(4)}' # cdm에서 조회하여 key값 지정

            key_modal_name = f'{cdm_key}_{value["execdate"]}_{value["exectime"]}_{value["source_modality"]}' # {person_id}_{orddate}(8)_{ordseqno}(4)_{execdate}(8)_{exectime}(6)_{modality}
            file_modal_path = os.path.join(save_path, value["source_modality"])
            file_modal_series_path = os.path.join(file_modal_path, key_modal_name, value["seriesdescription"]) # directory: {person_id}_{orddate}(8)_{ordseqno}(4)_{execdate}(8)_{exectime}(6)_{modality}/{Seriesdescription}
            new_filename = f'{cdm_key}_{value["execdate"]}_{seq[-9:]}.dcm' # rename한 dicom파일명 지정
                 
            for i in modal :
                if value["source_modality"] == i :
                    # make modality folder (이미 있으면 건너 뜀)
                    os.makedirs(os.path.join(file_modal_path), exist_ok=True)

                    # CR: make label, MR: make series folder
                    if value["source_modality"] == "CR":
                        label = labeling_xray(value)
                    # make series folder (이미 있으면 건너 뜀)
                    elif value["source_modality"] == 'MR':
                        # series구분 folder
                        os.makedirs(os.path.join(file_modal_series_path), exist_ok=True)
                        label = labeling_mr(value) 
                    elif value["source_modality"] == 'CT':
                        label = None

            # 파일명 수정하여 Dicom파일 저장
            if value["source_modality"] == "MR":
                new_filepath = os.path.join(file_modal_series_path, new_filename)
                value["rename_filepath"] = new_filepath
                img.save_as(new_filepath)
            else :
                new_filepath = os.path.join(file_modal_path, new_filename)
                value["rename_filepath"] = new_filepath
                img.save_as(new_filepath)
               
        ## DB에 저장할 데이터 쌓기
        value["label"] = label
        cleaned_value = tuple(None if pd.isna(x) else x for x in value[columns])
        result.append(cleaned_value) # (tuple(value[col] for col in columns))
        
        if len(result) >= batch_size:
            # cur.execute(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES " + ",".join(result))
            cur.executemany(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})", result)
            result = []
            conn.commit()
                
    if result :
        # cur.execute(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES " + ",".join(result))
        cur.executemany(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})", result)
        conn.commit()
        
    cur.close()
    conn.close()


def process_dicom(data_tuple):
    """
    단일 DICOM 파일을 처리하고 DB에 저장할 데이터 준비
    """
    value, save_path, modal, columns = data_tuple
    result = []
    new_filepath = None
    label = None
    source_filepath = value["source_filepath"]

    # 리스트나 딕셔너리를 튜플로 변환해 해시 가능하게 만듭니다.
    for key in value:
        if isinstance(value[key], list):
            value[key] = tuple(value[key])
        elif isinstance(value[key], dict):
            value[key] = tuple(sorted(value[key].items()))
        # numpy 데이터 타입을 파이썬 기본 타입으로 변환
        elif isinstance(value[key], (np.integer, np.floating)):  # numpy 타입이면
            value[key] = value[key].item()  # 파이썬 기본 타입으로 변환
            
    if pd.notna(value["person_id"]) and pd.notna(value["seriesdescription"]):
        try:
            img = pydicom.dcmread(source_filepath, stop_before_pixels=False)  # 태그 정보만 읽음
            seq = source_filepath.split('.dcm')[0]
            # CDM테이블의 key생성
            cdm_key = f'{int(value["person_id"])}_{value["orddate"]}_{str(int(value["ordseqno"])).zfill(4)}'
            key_modal_name = f'{cdm_key}_{value["execdate"]}_{value["exectime"]}_{value["source_modality"]}'
            file_modal_path = os.path.join(save_path, value["source_modality"])
            file_modal_series_path = os.path.join(file_modal_path, key_modal_name, value["seriesdescription"])
            ct_patientid_path = os.path.join(file_modal_path, value["patient_no"] + '_' + value["execdate"] + '_' + value["seriestime"])
            new_filename = f'{cdm_key}_{value["execdate"]}_{seq[-9:]}.dcm'

            if value["source_modality"] in modal:
                os.makedirs(os.path.join(file_modal_path), exist_ok=True)

                if value["source_modality"] == "CR":
                    label = labeling_xray(value)
                elif value["source_modality"] == 'MR':
                    os.makedirs(os.path.join(file_modal_series_path), exist_ok=True)
                    label = labeling_mr(value)
                elif value["source_modality"] == 'CT' and value['bodypartexamined'] in ['LUNG', 'CHEST', 'Chest Lung']:
                    os.makedirs(os.path.join(ct_patientid_path), exist_ok=True)
                    label = None

            if value["source_modality"] == "MR":
                new_filepath = os.path.join(file_modal_series_path, new_filename)
                value["rename_filepath"] = new_filepath
                img.save_as(new_filepath)
            elif value["source_modality"] == "CT":
                if value['bodypartexamined'] in ['LUNG', 'CHEST', 'Chest Lung']:
                    new_filepath = os.path.join(ct_patientid_path, new_filename)
                    value["rename_filepath"] = new_filepath
                    img.save_as(new_filepath)
            else:
                new_filepath = os.path.join(file_modal_path, new_filename)
                value["rename_filepath"] = new_filepath
                img.save_as(new_filepath)

        except Exception as e:
            print(f"Error processing file {source_filepath}: {e}")
            return []

    value["label"] = label
    # 튜플로 변환하여 결과 저장 (리스트를 튜플로 변환)
    cleaned_value = tuple(None if pd.isna(x) else (tuple(x) if isinstance(x, list) else x) for x in [value[col] for col in columns])
    result.append(cleaned_value)
    return result


def rename_dicom_multiprocessing(save_path, input_df, batch_size = 50000, max_workers=4, modal = ["CT", "MR", "CR"], dicom_cdm_table = "dicom_cdm_staging"):
    """
    dicom tag와 CDM을 결합한 데이터프레임을 이용해서 dicom 파일 폴더 및 파일명 수정
    생성되는 파일
    1. 파일명 변경된 이미지
    2. 원본파일명 to 변경된 파일명 이력 (DB에 저장)

    modality가 MR인 경우
    series description tag를 이용하여 series폴더 생성 후 dcm파일 분류
    
    modality가 XR인 경우
    XR폴더에 모든 파일 저장
    
    modality가 CT인 경우..
    2025.10 현재 CT폴더 내에 환자별 폴더를 기준으로 파일 저장
    2024.08 현재 아직 정해지지 않아 CT폴더에 모든 파일 저장
    """
    result = []
    schema_name = "your_schema"
    
    load_dotenv()
    
    # conn = postgres_connection()
    conn_pool = pgpool.SimpleConnectionPool(minconn=1, maxconn=10, dsn = f'dbname={os.environ.get("dbname")} user={os.environ.get("user")} password={os.environ.get("password")} host={os.environ.get("host")} port={os.environ.get("port")}')
    conn = conn_pool.getconn()
    cur = conn.cursor()
    
    # Dicom과 CDM이 연결된 테이블 만들기
    create_dicom_cdm(dicom_cdm_table = "dicom_cdm_ct")

    # 컬럼 이름 가져오기 (마지막 컬럼을 제외한 나머지)
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{dicom_cdm_table}' and table_schema = '{schema_name}'")
    columns = [col[0] for col in cur.fetchall()]

    # DataFrame의 각 행을 튜플로 변환하여 멀티프로세싱에 전달
    data_to_process = [(row._asdict(), save_path, modal, columns) for row in input_df.itertuples(index=False)]
    
    # 멀티프로세싱 사용
    with Pool(processes=max_workers) as pool:
        # tqdm을 사용해 진행 상황 표시
        results = list(tqdm(pool.imap(process_dicom, data_to_process), total=len(data_to_process), mininterval=60.0))

    # 결과를 하나로 합침
    for res in results:
        result.extend(res)

        if len(result) >= batch_size:
            cur.executemany(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) ON CONFLICT (source_filepath) DO NOTHING", result)
            result = []
            conn.commit()

    # 마지막 배치 처리
    if result:
        cur.executemany(f"INSERT INTO {schema_name}.{dicom_cdm_table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) ON CONFLICT (source_filepath) DO NOTHING", result)
        conn.commit()

    cur.close()
    conn.close()
    conn_pool.closeall()
    
def generate_date_list(start_date_str, end_date_str, date_format="%Y%m%d"):
    """
    시작일과 종료일 사이의 모든 날짜를 리스트로 생성합니다.

    Args:
        start_date_str (str): 시작일 문자열 (예: "20230926").
        end_date_str (str): 종료일 문자열 (예: "20240515").
        date_format (str): 날짜 형식.

    Returns:
        List[str]: "YYYYMMDD" 형식의 날짜 리스트.
    """
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    delta = end_date - start_date

    date_list = [(start_date + timedelta(days=i)).strftime(date_format) for i in range(delta.days + 1)]
    return date_list

## convert_nifti 추가 및 DB이용해서 labeling 결과 csv파일로 저장 기능 필요!
def convert_nifti(start_dir):
    """
    MRI Series별 dcm파일 nifti파일로 변환하기 위한 기능
    폴더 디렉토리 구조는 modality 디렉토리를 시작으로
    "./Key_dir/Series_dir/*.dcm" 으로 가정한다
    """
    folder_paths = glob(start_dir + "/*/*")

    for folder_path in folder_paths:
        input_directory = folder_path
        output_directory = folder_path
        dicom2nifti.convert_directory(input_directory, output_directory)


def print_execution_time(start_time, processname):
    logging.info(f"{processname} end, elapsed_time is : {datetime.now() - start_time}")

if __name__ == '__main__':
    start_time = datetime.now()
    
    # 로깅 설정을 main에서 먼저 설정
    import logging
    from datetime import datetime
    
    log_filename = f"log/main_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help = 'dicom파일이 저장된 경로 지정', default = './data')
    parser.add_argument('-d', help = '파일명을 변경한 dicom파일을 저장할 경로 지정', default = './output')

    args = parser.parse_args()

    ## Multilingual SBERT 모델 로드
    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 데이터로 저장할 Tag목록 정의
    tags = ["AccessionNumber", "StudyInstanceUID", "StudyDate", "StudyTime", "StudyDescription", "Manufacturer",
            "Modality", "SeriesDate", "SeriesTime", "SeriesInstanceUID", "BodyPartExamined",
            "Laterality", "PatientPosition", "SliceThickness", "Rows", "Columns",
            "WindowCenter", "WindowWidth", "SeriesDescription",  "KVP", "RequestedProcedureDescription",
            "ViewPosition", "ProtocolName", "ScanningSequence", "RepetitionTime", "AcquisitionContrast", "SeriesNumber",
            "InstanceNumber"
            , "ConvolutionKernel", "ImageType", "ImageOrientationPatient",
            "ContrastBolusAgent", "ContrastBolusVolume", "ContrastFlowRate", "ContrastFlowDuration", "ProcedureCodeSequence"
            ]
    start_date_str = "20230926" # "20230926" # "20240516"
    end_date_str = "20231130" # "20240515" # "20240531"
    date_list = generate_date_list(start_date_str, end_date_str)
    
    try :
        logging.info(f"Starting DICOM processing for date range: {start_date_str} to {end_date_str}")
        logging.info(f"Total dates to process: {len(date_list)}")
        
        for workdate in date_list:
            logging.info(f"Processing workdate: {workdate}")
            ## dicom파일명과 태그정보 수집
            basic_info, tag_info = load_dicom_tags_multiprocessing(args.s, tags, int(workdate), int(workdate), modal = ['CT'] ) # modal = ['CR', 'MR', 'CT']
            print_execution_time(start_time, 'load_dicom_tags')

            ## 수집한 dicom파일관련 정보들을 DB에 저장
            save_dicom_tags_to_postgres(basic_info) # 속도 문제로 중복되는 데이터가 들어갈 수 있으니 주의 필요 
            print_execution_time(start_time, 'save_dicom_key_source_to_postgres')
            
            # save_dicom_tags_source(tag_info)
            # print_execution_time(start_time, 'save_dicom_tags_to_postgres')
            if not tag_info.empty:
                pass # 추가된 데이터 없으므로 넘어가기
                # save_tag_info_to_parquet(tag_info, './output/tag_info_temp', workdate)
                # print_execution_time(start_time, 'save_dicom_tags_to_parquet')
            
            
            # # procedure_occurrence_embedding 테이블에 데이터 넣기
            # add_procedure_embedding(model)
            # print_execution_time(start_time, 'add_procedure_embedding')
            
            ## dicom파일관련 정보들을 CDM데이터와 연결한 데이터 DB에 저장
            merged_df = search_in_CDM(workdate, workdate) # output -> dicom정보와 CDM매핑키
            print_execution_time(start_time, 'search_in_CDM')

            # merged_df = custom_search_in_CDM()
            # ## 연결한 CDM데이터를 기반으로 dicom파일명 변경하여 저장
            # # rename_dicom(args.d, merged_df)
            rename_dicom_multiprocessing(args.d, merged_df, modal = ['CT'], dicom_cdm_table = "dicom_cdm_ct") # modal = ['CR', 'MR', 'CT']
            print_execution_time(start_time, 'rename_dicom')
            
            # # 의도치 않게 삭제되는 row가 있을 수 있으니 데이터를 확인하고, 테스트 후 사용해야함
            # # delete_duplicate_row(start_date, end_date)
            # # print_execution_time(start_time, 'delete_duplicate_row')
          
    except Exception as e : 
        print_execution_time(start_time, f'Execution failed: {e}')