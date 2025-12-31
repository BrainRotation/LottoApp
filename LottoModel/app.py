import os
import sys

# 현재 디렉토리를 시스템 경로에 추가 (모듈 import 문제 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from flask import Flask, jsonify
from lottomodel import LottoPredictor
import pandas as pd

app = Flask(__name__)

# 모델 전역 변수
model_instance = None
recent_data_cache = None

# 파일 경로 설정 (AWS 등 배포 환경 호환)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'lotto-1052.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'lotto_lstm.pth')

def init_model():
    global model_instance, recent_data_cache
    print("모델 초기화 중...")
    
    # 1. 모델 인스턴스 생성
    # 데이터 파일 존재 확인
    if not os.path.exists(DATA_PATH):
        print(f"경고: 데이터 파일({DATA_PATH})을 찾을 수 없습니다.")
        # 빈 모델이라도 생성 시도
    
    try:
        model_instance = LottoPredictor(data_path=DATA_PATH, window_size=20)
        
        # 학습된 모델 파일 로드
        if os.path.exists(MODEL_PATH):
            model_instance.load_model(MODEL_PATH)
            print("학습된 모델 로드 완료")
        else:
            print("경고: 학습된 모델 파일(lotto_lstm.pth)이 없습니다. 예측이 랜덤하게 동작할 수 있습니다.")
    except Exception as e:
        print(f"모델 생성 중 치명적 오류: {e}")

    # 2. 최근 데이터 캐싱
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # 컬럼 처리 로직 유지
            if 'num1' in df.columns:
                cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
            else:
                cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']
            recent_data_cache = df[cols].values[-20:]
            print("최근 데이터 캐싱 완료")
        else:
             print("데이터 파일 부재로 캐싱 실패")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")

# 앱 시작 시 초기화
init_model()

@app.route('/', methods=['GET'])
def health_check():
    """AWS 로드밸런서 상태 확인용"""
    return "Lotto API is running", 200

@app.route('/predict', methods=['GET'])
def predict():
    """
    플러터 앱에서 호출할 API 엔드포인트
    """
    global model_instance, recent_data_cache
    
    if model_instance is None or recent_data_cache is None:
        return jsonify({"error": "Model not initialized"}), 500
    
    try:
        # 모델 예측 실행
        recommended_numbers = model_instance.predict(recent_data_cache)
        
        # JSON 형태로 응답 (회차는 Flutter 앱에서 자체 계산)
        return jsonify({
            "status": "success",
            "numbers": recommended_numbers,
            "message": "AI 추천 번호 생성 완료"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 서버 설정
    # host='0.0.0.0': 외부에서 접속 가능 (폰, 에뮬레이터 등)
    # port=5000: Flask 기본 포트
    HOST = '0.0.0.0'
    PORT = 5000
    
    print(f"\n{'='*50}")
    print(f"🚀 Lotto AI 서버 시작")
    print(f"{'='*50}")
    print(f"📍 로컬 접속: http://127.0.0.1:{PORT}")
    print(f"📍 네트워크 접속: http://[내부IP]:{PORT}")
    print(f"{'='*50}\n")
    
    app.run(host=HOST, port=PORT, debug=False)

