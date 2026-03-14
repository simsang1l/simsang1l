import dicom2nifti
import pydicom

from glob import glob
import os
import pandas as pd
from datetime import datetime
import multiprocessing
from tqdm import tqdm

def convert_nifti(start_dir):
    """
    MRI Series별 dcm파일 nifti파일로 변환하기 위한 기능
    폴더 디렉토리 구조는 modality 디렉토리를 시작으로
    "./Key_dir/Series_dir/*.dcm" 으로 가정한다
    """
    folder_paths = glob(start_dir + "/*/*")

    for folder_path in tqdm(folder_paths, mininterval=60):
        input_directory = folder_path
        output_directory = folder_path
        dicom2nifti.convert_directory(input_directory, output_directory)

def convert_single_folder(folder_path):
    """
    단일 폴더를 DICOM에서 NIfTI로 변환
    """
    try:
        # DICOM 파일이 있는지 확인
        dcm_files = glob(os.path.join(folder_path, "*.dcm"))
        if not dcm_files:
            print(f"No DICOM files found in {folder_path}. Skipping...")
            return False

        # 출력 디렉토리 설정 (입력 디렉토리와 동일하게 설정)
        output_directory = folder_path

        # 변환 수행
        dicom2nifti.convert_directory(folder_path, output_directory)
        print(f"Successfully converted: {folder_path}")
        return True
    except Exception as e:
        print(f"Error converting {folder_path}: {e}")
        return False

def convert_nifti_parallel(start_dir):
    """
    병렬로 DICOM 파일을 NIfTI로 변환
    """
    folder_paths = glob(start_dir + "/*/*")  # Series 디렉토리 탐색

    # 진행률 표시와 병렬 처리
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(convert_single_folder, folder_paths),
            total=len(folder_paths),
            desc="Converting folders",
            unit="folder",
            mininterval=60.0
        ))

    # 변환 결과 요약
    successful_conversions = sum(results)
    print(f"\nTotal folders: {len(folder_paths)}")
    print(f"Successfully converted: {successful_conversions}")
    print(f"Failed or skipped: {len(folder_paths) - successful_conversions}")


def labeling_nifti(start_dir):
    """
    국립암센터 기준으로 labeling (and 조건)
    T1ce:
        - ScanningSequence = 'GR'
        - RepetitionTime <= 1000
        - SeriesDescription에 TOF, mIP, reformat은 제외
    bb:
        - ScanningSequence = "IR"
        - RepetitionTime >= 2000
        - RepetitionTime <= 3000
        - "irFSE" in SeriesDescription
    Others:
        - RepetitionTime >= 4000
        - AcquisitionConstrast != "T1"
        - "t1" not in SeriesDescription

    충북대 labeling 기준
    t1ce: T1과 CE둘 다 있는 경우
    bb: bb가 포함된 경우?
    other: 나머지
    """
    filelist = []

    for root, directories, files in os.walk(start_dir):
        if len(directories) == 0:
            nifti_filelist = [_ for _ in files if _.endswith('.nii.gz')]
            niftifile_path = None

            scanningsequence = None
            repetitiontime = 0
            seriesdescription = None
            acquisitioncontrast = ''
            label = None
            
            # dicom파일에서 tag 추출 및 레이블링
            for file in files:
                if file.endswith(".dcm") :
                    dicomfile_path = os.path.join(root, file)
                    dc = pydicom.dcmread(dicomfile_path)

                    if "ScanningSequence" in dc :
                        scanningsequence = dc.ScanningSequence
                    if "RepetitionTime" in dc :
                        repetitiontime = dc.RepetitionTime
                    seriesdescription = dc.SeriesDescription
                    if "AcquisitionContrast" in dc:
                        acquisitioncontrast = dc.AcquisitionContrast

                    # label 설정
                    if (scanningsequence == 'GR' and repetitiontime <= 1000 and ("tof" not in seriesdescription.lower() and "mip" not in seriesdescription.lower() and 'reformat' not in seriesdescription.lower())) or ("t1" in seriesdescription.lower() and "ce" in seriesdescription.lower() and "bb" not in seriesdescription.lower()):
                        label = "t1ce"
                    elif (scanningsequence == 'IR' and repetitiontime >= 2000 and repetitiontime <= 3000 and "irfse" in seriesdescription) or 'bb' in seriesdescription.lower():
                        label = "bb"
                    elif repetitiontime >= 4000 and "t1" not in acquisitioncontrast.lower() and "t1" not in seriesdescription.lower():
                        label = "Others"
                    else:
                        label = "Others"
                    
                    break

            # nifti파일 레이블링 결과 저장
            for nifti in nifti_filelist:
                niftifile_path = os.path.join(root, nifti)
                filelist.append([niftifile_path, scanningsequence, repetitiontime, seriesdescription, acquisitioncontrast, label])

    filelist = pd.DataFrame(filelist)

    filelist.columns = ["filepath", "scanningsequence", "repetitiontime", "seriesdescription", "acquisitioncontrast", "label"]
    filelist.to_csv("./label/mri_label_org.csv", index = False)
    mri_label = filelist[["filepath", "label"]]
    mri_label.to_csv('./label/mri_label.csv', index = False)

from datetime import datetime
import argparse

def print_execution_time(start_time, processname):
    print(f"{processname} end, elapsed_time is : {datetime.now() - start_time}")

if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help = 'MRI데이터가 저장된 모달리티 폴더입력', default = './data/MR')

    args = parser.parse_args()
    # convert_nifti(args.s)
    convert_nifti_parallel(args.s)
    
    lst = glob(args.s + "/*/*/*")
    # print(lst)
    cnt = 0
    for i in lst :
        if '.nii.gz' in i:
            cnt += 1
        if cnt == 1 :
            print(i)
    print(cnt)
    
    print_execution_time(start_time, 'convert_nifti')
    # labeling_nifti(args.s)
    print_execution_time(start_time, 'labeling_nifti')

    