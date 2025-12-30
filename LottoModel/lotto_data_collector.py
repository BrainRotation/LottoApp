import requests
import pandas as pd
import time
# from tqdm import tqdm
import os
import sys

print("스크립트 시작", flush=True)

def collect_lotto_data(start_drw=1, save_path='data/lotto_history.csv'):
    print("로또 데이터 수집을 시작합니다...", flush=True)
    
    # 데이터 저장할 리스트
    lotto_data = []
    
    drw_no = start_drw
    
    # 진행 상황 표시를 위해 tqdm 사용 (총 회차를 모르므로 일단 무한루프 + break)
    # 대략 1100회 넘었으므로 1200 정도로 range 잡고 진행해도 됨
    
    # pbar = tqdm(range(start_drw, 2000), desc="데이터 수집 중")
    
    for drw_no in range(start_drw, 2000):
        if drw_no % 100 == 0:
            print(f"{drw_no}회차 수집 중...", flush=True)
        
        url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drw_no}'
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['returnValue'] == 'fail':
                print(f"\n{drw_no}회차에서 데이터 수집 종료 (마지막 회차 도달)")
                break
                
            lotto_dict = {
                'drwNo': data['drwNo'], # 회차
                'drwNoDate': data['drwNoDate'], # 날짜
                'drwtNo1': data['drwtNo1'],
                'drwtNo2': data['drwtNo2'],
                'drwtNo3': data['drwtNo3'],
                'drwtNo4': data['drwtNo4'],
                'drwtNo5': data['drwtNo5'],
                'drwtNo6': data['drwtNo6'],
                'bnusNo': data['bnusNo'] # 보너스 번호
            }
            
            lotto_data.append(lotto_dict)
            
            # 서버 부하 방지를 위해 약간의 딜레이
            # time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError at {drw_no}: {e}")
            break
            
    if not lotto_data:
        print("수집된 데이터가 없습니다.")
        return

    # DataFrame 생성 및 저장
    df = pd.DataFrame(lotto_data)
    
    # data 폴더가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\n데이터 수집 완료! 총 {len(df)}개 회차 저장됨.")
    print(f"저장 경로: {save_path}")
    print(df.tail())

if __name__ == "__main__":
    collect_lotto_data()
