import pandas as pd
import os

file_path = 'data/lotto-1052.csv.xlsx'

try:
    # 엑셀 파일로 시도
    print("엑셀 파일로 읽기 시도...")
    df = pd.read_excel(file_path)
    print("성공 (Excel)")
except Exception as e:
    print(f"엑셀 읽기 실패: {e}")
    try:
        # CSV 파일로 시도
        print("CSV 파일로 읽기 시도...")
        df = pd.read_csv(file_path)
        print("성공 (CSV)")
    except Exception as e2:
        print(f"CSV 읽기 실패: {e2}")
        exit()

print(f"데이터 크기: {df.shape}")
print("컬럼 목록:", df.columns.tolist())
print(df.head())

