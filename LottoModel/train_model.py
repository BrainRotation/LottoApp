from lottomodel import LottoPredictor # 별칭 사용
import torch
import numpy as np
import pandas as pd

def train():
    print("=== 딥러닝 모델 학습 시작 (LSTM + Multi-label Classification) ===")
    
    # 1. 모델 인스턴스 생성
    # window_size=20: 과거 20회차 데이터를 보고 다음 회차 예측
    dl_model = LottoPredictor(data_path='data/lotto-1052.csv', window_size=20)
    
    # 2. 모델 학습
    dl_model.train(epochs=300, batch_size=64, lr=0.001)
    
    # 3. 모델 저장
    dl_model.save_model('lotto_lstm.pth')
    
    # 4. 정확도 평가
    dl_model.calculate_accuracy(test_size=50) # 최근 50회차로 성능 테스트
    
    print("\n=== 예측 테스트 ===")
    
    # 최근 데이터를 가져와서 다음 회차 예측
    df = pd.read_csv('data/lotto-1052.csv')
    
    if 'num1' in df.columns:
        cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
    else:
        cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']
        
    recent_data = df[cols].values[-20:] # 마지막 20개 회차
    
    print("최근 20회차 데이터 로드 완료.")
    
    # 5번 정도 예측 실행
    print("\n[추천 번호]")
    for i in range(5):
        pred = dl_model.predict(recent_data)
        print(f"추천 {i+1}: {pred}")

if __name__ == "__main__":
    train()