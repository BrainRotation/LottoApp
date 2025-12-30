import time
from lottomodel import LottoPredictor
import pandas as pd
import numpy as np

def benchmark():
    print("성능 측정을 시작합니다...")
    
    # 모델 로드 (시간 측정 제외)
    model = LottoPredictor(data_path='data/lotto-1052.csv', window_size=20)
    model.load_model('lotto_lstm.pth')
    
    # 데이터 준비
    df = pd.read_csv('data/lotto-1052.csv')
    if 'num1' in df.columns:
        cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
    else:
        cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']
    recent_data = df[cols].values[-20:]
    
    # 워밍업 (첫 실행은 캐싱 등으로 느릴 수 있음)
    model.predict(recent_data)
    
    # 측정 시작
    iterations = 100
    times = []
    
    print(f"\n{iterations}회 반복 추론 중...")
    
    for _ in range(iterations):
        start_time = time.perf_counter() # 정밀 시간 측정
        
        _ = model.predict(recent_data)
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000) # ms 단위 변환
        
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n=== 성능 측정 결과 ===")
    print(f"평균 실행 시간: {avg_time:.4f} ms")
    print(f"최소 실행 시간: {min_time:.4f} ms")
    print(f"최대 실행 시간: {max_time:.4f} ms")

if __name__ == "__main__":
    benchmark()

