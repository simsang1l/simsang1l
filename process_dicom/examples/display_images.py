import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
from PIL import Image, ImageOps
from glob import glob

def xray_cm_to_pivot(df):
    # transform confusion_matrix to pivot using xray data

    y_true = df["label"]
    y_pred = df["pred"]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    total_by_actual_label = cm.sum(axis = 1)
    print(total_by_actual_label)


    cm_df = pd.DataFrame(cm, index = np.unique(y_true), columns = np.unique(y_pred))
    print(cm_df)

    df_melt = cm_df.reset_index().melt(id_vars = ["index"])
    df_melt.rename(columns = {"index":"label", "variable":"pred", "value":"count"}, inplace = True)
    # print( df_melt.groupby('label')['count'].sum() )
    df_melt["total"] = df_melt.groupby('label')['count'].transform("sum")
    df_melt["ratio"] = df_melt["count"] / df_melt["total"]
    # df_melt.to_csv('./result/result_melt.csv',index = False)

    pivot_table = df_melt.pivot_table(index = "label", columns = "pred", values=["count", "ratio"], fill_value = 0)
    print(pivot_table)
    # pivot_table.to_csv("./result/xray_result_pivot.csv")


def load_result(df, label):
    path = "./data/CR"
    df = df[df["label"] == label]
    df["new_filepath"] = df["new_filepath"].apply(lambda x : os.path.join(path, x))

    return df

def display_one_dicom_image(image_path):
    ds = pydicom.dcmread(image_path)
    plt.imshow(ds.pixel_array, cmap='gray')
    plt.axis('off')
    plt.show()

def display_dicom_image(image_path, n):
    fig, axs = plt.subplots(1, n, figsize = (n * 5, 5))
    for ax, dicom_path in zip(axs, image_path):
        print(dicom_path)
        ds = pydicom.dcmread(dicom_path)
        ax.imshow(ds.pixel_array, cmap='gray')
        ax.axis('off')
    plt.show()

def display_images_on_click(csv_path, label, n = 3):
    """이미지를 클릭했을 때 실행되는 함수"""
    df = load_result(csv_path, label)
    paths = df["new_filepath"].tolist()

    for i in range(0, len(paths), n):
        clear_output(wait=True)  # 이전 이미지 지우기
        display_dicom_image(paths[i:i+n], n)
        input("Press Enter to continue...")  # 사용자 입력 대기
        

    # current_image_index[0] += 1  # 다음 이미지로 인덱스 증가
    # if current_image_index[0] < len(dicom_paths):
    #     plt.clf()  # 현재 이미지 클리어
    #     load_and_display_xray(dicom_paths[current_image_index[0]])
    # else:
    #     print("No more images.")

def dicom_to_pil_image(dicom_path):
    """DICOM 파일을 읽어 PIL 이미지로 변환합니다."""
    dicom_data = pydicom.dcmread(dicom_path)
    image_array = dicom_data.pixel_array
    # print(dicom_data.Rows, dicom_data.Columns)
    # print(image_array.shape)
    # 이미지 스케일 조정 및 8비트 변환
    # image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    # image_array = np.uint8(image_array)
    # print(image_array.shape)
    pil_image = Image.fromarray(image_array)
    return pil_image

def analyze_and_display_dicom_images(dicom_paths):
    """여러 DICOM 이미지를 분석하여 원본 및 조건부 역상 처리된 이미지를 표시합니다."""
    for path in dicom_paths:
        img = dicom_to_pil_image(path)
        
        # 배경색 판단 및 역상 처리 여부 결정
        pixels = img.load()
        corners = [pixels[0, 0], pixels[0, img.height - 1], pixels[img.width - 1, 0], pixels[img.width - 1, img.height - 1]]
        avg_corner_brightness = sum(corners) / len(corners)
        print('avg_corner_brightness;;', avg_corner_brightness)
        background_color = 'white' if avg_corner_brightness > 150 else 'black'
        
        # 역상 처리
        if background_color == 'white':
            inverted_img = ImageOps.invert(img)
        else:
            inverted_img = img  # 배경이 검정색일 경우, 역상 처리 없이 원본 이미지 사용
        
        # 이미지 표시
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(inverted_img, cmap='gray')
        ax[1].set_title('Processed Image')
        ax[1].axis('off')
        
        plt.show()

def analyze_and_display_two_dicom_images(dicom_paths, inverted_path):
    """여러 DICOM 이미지를 분석하여 원본 및 조건부 역상 처리된 이미지를 표시합니다."""
    for path, i_path in zip(dicom_paths, inverted_path):
        img = dicom_to_pil_image(path)
        inverted_img = dicom_to_pil_image(i_path)
        
        # 이미지 표시
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(inverted_img, cmap='gray')
        ax[1].set_title('Processed Image')
        ax[1].axis('off')
        
        plt.show()


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
    
    print('corners;;', corners)
    print('avg_brightness;;', avg_corner_brightness)
    print('threshold;;', threshold)
    # return avg_brightness > 127  # 임계값은 조정 가능
    return avg_corner_brightness > threshold

def _is_image_inverted(pixel_array):
    histogram, _ = np.histogram(pixel_array.flatten(), bins=256, range=(0, 256))
    dark_pixels = np.sum(histogram[:128])  # 어두운 픽셀 수
    bright_pixels = np.sum(histogram[128:])  # 밝은 픽셀 수
    print(dark_pixels, bright_pixels)
    return bright_pixels > dark_pixels

def process_and_show_dicom(original_dicom_path):
    """DICOM 이미지를 처리하고 저장하는 함수."""
    invert_result = []
    for path in original_dicom_path:
        inverted = 0
        # 원본 DICOM 파일 읽기
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array

        filename = path.split('/')[-1]
        # 이미지가 역상인지 확인
        if is_image_inverted(pixel_array):
            inverted = 1
            # 역상 처리를 적용
            inverted_array = np.invert(pixel_array)
            inverted_array = inverted_array.astype(ds.pixel_array.dtype)
            ds.PixelData = inverted_array.tobytes()
            # ds.save_as(os.path.join(new_dicom_path, filename))
        else :
            inverted_array = pixel_array
        
        # invert_result.append([os.path.join(new_dicom_path, filename), inverted])
        # print(path, filename)
        # print(pixel_array.shape, inverted_array.shape)
        img = dicom_to_pil_image(path)
        # print(dicom_data.Rows, dicom_data.Columns)
        # print(image_array.shape)
        # 이미지 스케일 조정 및 8비트 변환
        # image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
        # image_array = np.uint8(image_array)
        # print(image_array.shape)
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


if __name__ == "__main__":
    df = pd.read_csv("./result/_xray_result.csv")
    label = "Others"
    display_images_on_click(df, label)
    # dicom_to_pil_image(glob("./data/CR/*")[0])
    # display_one_dicom_image(glob("./data/CR/*"))
    # analyze_and_display_dicom_images(glob("./data/CR_inverted/*"))
    # analyze_and_display_two_dicom_images(glob("./data/CR/*"), glob("./data/CR_inverted/*"))
    # process_and_show_dicom(glob("./data/CR/*"))

    
