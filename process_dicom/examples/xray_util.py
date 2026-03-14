import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageOps
import pydicom
import os
from glob import glob
import matplotlib.pyplot as plt

def get_dicom_metadata(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    metadata = {
        'filepath': os.path.basename(dicom_path),
        'ViewPosition': ds.ViewPosition if 'ViewPosition' in ds else '',
        'SeriesDescription': ds.SeriesDescription if 'SeriesDescription' in ds else '',
        'ProtocolName': ds.ProtocolName if 'ProtocolName' in ds else '',
        'StudyDescription': ds.StudyDescription if 'StudyDescription' in ds else ''
    }
    return metadata

def determine_label(metadata):
    # Convert all fields to uppercase for case-insensitive comparison
    view_position = metadata['ViewPosition'].upper()
    series_description = metadata['SeriesDescription'].upper()
    protocol_name = metadata['ProtocolName'].upper()
    study_description = metadata['StudyDescription'].upper()

    # AP conditions
    if 'AP' in view_position:
        return 'AP'
    if 'AP' in series_description or 'AP' in protocol_name:
        return 'AP'
    if not series_description and not protocol_name and 'AP' in study_description:
        return 'AP'
    
    # PA conditions
    if 'PA' in view_position:
        return 'PA'
    if 'PA' in series_description or 'PA' in protocol_name:
        return 'PA'
    if not series_description and not protocol_name and 'PA' in study_description:
        return 'PA'
    
    # LATERAL conditions
    if 'LAT' in view_position:
        return 'Lateral'
    if 'LAT' in series_description or 'LAT' in protocol_name:
        return 'Lateral'
    if not series_description and not protocol_name and 'LAT' in study_description:
        return 'Lateral'
    
    # Default to Others
    return 'Others'

def label_xray(df, execute_time):
    """
    Xray 라벨링
    """
    data = []
    df = df[df["modality"] == "CR"]
    df = df.drop(columns = ["label"])
    
    
    for i in range(len(df)):
        filepath = df.iloc[i]["new_filepath"]
        metadata = get_dicom_metadata(filepath)
        metadata["filepath"] = filepath
        label = determine_label(metadata)
        metadata['label'] = label
        data.append(metadata)
        
    df = pd.DataFrame(data)

    df.to_csv(f"./label/xray_label_{execute_time}.csv", index = False)


def _label_xray(df):
    """
    studydescription과 protocolname으로 라벨링
    """

    df = df[df["modality"] == "CR"]
    df = df.drop(columns = ["label"])

    df["label"] = "Others"
    ap_condition = df["seriesdescription"].str.contains("AP", case=False, na=False) | df["study_description"].str.contains("AP", case=False, na=False)
    pa_condition = df["seriesdescription"].str.contains("PA", case=False, na=False) | df["study_description"].str.contains("PA", case=False, na=False)
    lat_condition = df["seriesdescription"].str.contains("LAT", case=False, na=False) | df["study_description"].str.contains("LAT", case=False, na=False)

    df.loc[ap_condition, "label"] = "AP"
    df.loc[pa_condition, "label"] = "PA"
    df.loc[lat_condition, "label"] = "Lateral"

    df.to_csv("./label/_xray_label.csv", index = False)


def xray_metric(label_dir, pred_dir, execute_time):
    label_file = pd.read_csv(label_dir)
    # label_file = label_file[label_file["modality"] == 'CR']
    label_file = label_file[["filepath", "label"]]
    label_file.sort_values("filepath")
    label_file["filepath"] = label_file["filepath"].apply(lambda x: x.split('/')[-1])
    # label_file.loc[label_file["label"] == "others", "label"] = "Others"

    pred_file = pd.read_csv(pred_dir)
    pred_file.sort_values("file")
    pred_file = pred_file[["file", "pred"]]

    result = pd.merge(label_file, pred_file, left_on="filepath", right_on = "file", how = "inner")
    result.to_csv(f"result/xray_result_{execute_time}.csv", index = False)

    label = result[["label"]]
    pred = result[["pred"]]

    num_classes = len(np.unique(pred))

    accuracy = accuracy_score(label, pred)
    recall = recall_score(label, pred, average = 'macro')
    precision = precision_score(label, pred, average = 'macro')

    specificities = []
    
    encoder = LabelEncoder()
    actual_encoded = encoder.fit_transform(label)
    predicted_encoded = encoder.transform(pred)
    for i in range(num_classes):
        tn = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) != i))
        fp = np.sum((np.array(actual_encoded) != i) & (np.array(predicted_encoded) == i))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    metric_report = pd.DataFrame([[accuracy, recall, precision, sum(specificities) / len(specificities)]], columns = ["accuracy", "recall", "precision", "specificity"])
    metric_report.to_csv(f"result/xray_metric_{execute_time}.csv", index = False)


def dicom_to_pil_image(dicom_path):
    """DICOM 파일을 읽어 PIL 이미지로 변환합니다."""
    dicom_data = pydicom.dcmread(dicom_path, force=True)
    image_array = dicom_data.pixel_array
    
    # 이미지 스케일 조정 및 8비트 변환
    # image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    # image_array = np.uint8(image_array)
    
    pil_image = Image.fromarray(image_array)
    return pil_image, dicom_data

def save_pil_image_as_dicom(pil_img_array, dicom_data, output_dicom_path):

    # DICOM 파일에 이미지 데이터 설정
    dicom_data.PixelData = pil_img_array.tobytes()

    # Rows와 Columns 업데이트
    dicom_data.Rows, dicom_data.Columns = pil_img_array.shape[:2]

    # 기타 필요한 DICOM 태그 설정/수정
    # 예: ds.PatientName = "Your Patient Name"

    # 새 DICOM 파일로 저장
    dicom_data.save_as(output_dicom_path)


def invert_dicom_images(dicom_paths, output_path):
    """여러 DICOM 이미지를 분석하여 원본 및 조건부 역상 처리된 이미지를 표시합니다."""
    invert_result = []

    for path in dicom_paths:
        print(path)
        img, dicom_data = dicom_to_pil_image(path)
        
        # 배경색 판단 및 역상 처리 여부 결정
        pixels = img.load()
        corners = [pixels[0, 0], pixels[0, img.height - 1], pixels[img.width - 1, 0], pixels[img.width - 1, img.height - 1]]
        avg_corner_brightness = sum(corners) / len(corners)
        max_pixel_value = img.getextrema()[1]
        background_color = 'white' if avg_corner_brightness > max_pixel_value / 2 else 'black'
        
        # 역상 처리
        if background_color == 'white':
            inverted_img = ImageOps.invert(img)
            invert_result.append([path.split('/')[-1], 1])
        else:
            inverted_img = img
            invert_result.append([path.split('/')[-1], 0])
        
        save_pil_image_as_dicom(np.array(inverted_img), dicom_data, os.path.join(output_path, path.split('/')[-1]))

    pd.DataFrame(invert_result, columns=["파일명", "역상여부"]).to_csv(os.path.join(output_path, "invert_result.csv"))

def is_image_inverted(pixel_array):
    """이미지가 역상인지 확인하는 함수. 역상이라고 판단되면 True를 반환합니다."""
    # 모서리 픽셀의 밝기를 샘플링하여 평균 값을 계산
    corners = [
        pixel_array[0, 0], pixel_array[0, -1], 
        pixel_array[-1, 0], pixel_array[-1, -1]
    ]
    max_pixel_value = np.max(pixel_array)
    avg_corner_brightness = np.mean(corners)
    threshold = max_pixel_value / 2  # 비트 깊이에 따른 동적 임계값 설정
    
    return avg_corner_brightness > threshold

def process_and_save_dicom(original_dicom_path, new_dicom_path, execute_time):
    """DICOM 이미지를 처리하고 저장하는 함수."""
    invert_result = []
    for path in original_dicom_path:
        inverted = 0
        # 원본 DICOM 파일 읽기
        ds = pydicom.dcmread(path)
        image_array = ds.pixel_array

        # Convert to 8-bit grayscale
        pixel_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)

        filename = path.split('/')[-1]
        # 이미지가 역상인지 확인
        if is_image_inverted(pixel_array):
            inverted = 1
            # 역상 처리를 적용
            inverted_array = np.invert(pixel_array)
            inverted_array = inverted_array.astype(ds.pixel_array.dtype)
            ds.PixelData = inverted_array.tobytes()
            ds.save_as(os.path.join(new_dicom_path, filename))
        else :
            ds.save_as(os.path.join(new_dicom_path, filename))
        
        invert_result.append([os.path.join(new_dicom_path, filename), inverted])
    
    pd.DataFrame(invert_result, columns=["파일명", "역상여부"]).to_csv(os.path.join(new_dicom_path, f"./result/invert_result_{execute_time}.csv"))


def process_and_show_dicom(original_dicom_path):
    """DICOM 이미지를 처리하고 저장하는 함수."""
    invert_result = []
    for path in original_dicom_path:
        inverted = 0
        # 원본 DICOM 파일 읽기
        ds = pydicom.dcmread(path)
        image_array = ds.pixel_array

        # Convert to 8-bit grayscale
        pixel_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)

        filename = path.split('/')[-1]
        # 이미지가 역상인지 확인
        if is_image_inverted(pixel_array):
            inverted = 1
            # 역상 처리를 적용
            inverted_array = np.invert(pixel_array)
            inverted_array = inverted_array.astype(ds.pixel_array.dtype)
            ds.PixelData = inverted_array.tobytes()
            print("역상임")
            # ds.save_as(os.path.join(new_dicom_path, filename))
        else :
            inverted_array = pixel_array
            print("역상아님")

        img = dicom_to_pil_image(path)
        inverted_img = Image.fromarray(inverted_array)
        
        # 이미지 표시
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(inverted_img, cmap='gray')
        ax[1].set_title('Processed Image')
        ax[1].axis('off')
        
        plt.show()


def extract_xr_from_result_report(file_path):
    df = pd.read_csv(file_path)
    df = df[df["modality"].isin(["CR"])]

    df.to_csv('./result/CR_result_report.csv', index = False)

if __name__ == "__main__":
    execute_time = datetime.now().strftime('%Y%m%d')

    df = pd.read_csv("./data/result_report.csv", dtype=str)
    label_xray(df, execute_time)

    xray_label_dir = "./label/xray_label.csv"
    xray_pred_dir = "./data/CR/result.csv"
    xray_metric(xray_label_dir, xray_pred_dir)

    

