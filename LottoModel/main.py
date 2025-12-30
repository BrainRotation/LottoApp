"""
로또 모델 로컬 테스트 스크립트 (서버 아님!)

이 파일은 Flask 서버(app.py)와 무관하게 로컬에서 모델을 테스트하는 용도입니다.
실제 서버 실행은 app.py를 사용하세요: python app.py
"""

import os
import pandas as pd
from lotto_data_collector import get_lotto_numbers
from lottomodel import LottoPredictor

def main():
    # 1. 데이터 확인 및 수집
    data_path = 'data/lotto_history.csv'
    if not os.path.exists(data_path):
        print("데이터 파일이 없습니다. 최신 로또 데이터를 수집합니다...")
        df = get_lotto_numbers()
        if not os.path.exists('data'):
            os.makedirs('data')
        df.to_csv(data_path, index=False, encoding='utf-8-sig')
        print("데이터 수집 완료.")
    else:
        print("기존 데이터 파일을 확인했습니다.")

    # 2. 모델 준비 (학습 또는 로드)
    predictor = LottoPredictor(data_path=data_path, window_size=5)
    model_path = 'lotto_model.pth'
    
    if os.path.exists(model_path):
        print("저장된 모델을 불러옵니다...")
        predictor.load_saved_model(model_path)
        # 데이터 로드는 필요 없지만, 예측을 위해 최근 데이터가 필요하므로 로드
        predictor.load_data() 
    else:
        print("새로운 모델을 학습합니다...")
        predictor.load_data()
        predictor.preprocess()
        predictor.build_model()
        predictor.train(epochs=100)
        predictor.save_model(model_path)

    # 3. 다음 회차 번호 예측
    # 전체 데이터 중 가장 마지막 5회차(window_size)를 가져옴
    last_n_rows = predictor.rows[-predictor.window_size:]
    
    print("\n" + "="*50)
    print("🔮 AI 로또 번호 추천 🔮")
    print("="*50)
    
    # 5세트 추천
    for i in range(5):
        nums = predictor.predict_next(last_n_rows)
        print(f"추천 조합 {i+1}: {nums}")
    
    print("="*50)

if __name__ == "__main__":
    main()

