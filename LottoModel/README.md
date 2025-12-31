# 🎱 Lotto AI - LSTM 딥러닝 모델 서버

로또 번호를 예측하는 LSTM 딥러닝 모델 기반 Flask API 서버

## 📁 파일 구조

```
LottoModel/
├── app.py                    # Flask API 서버 (메인)
├── lottomodel.py             # LSTM 모델 클래스
├── lotto_lstm.pth            # 학습된 모델 파일
├── train_model.py            # 모델 학습 스크립트
├── lotto_data_collector.py   # 로또 데이터 수집
├── requirements.txt          # 필요 라이브러리
├── .gitignore               # Git 제외 파일
└── data/
    └── lotto-1052.csv       # 학습 데이터
```

## 🚀 설치 및 실행

### 1. 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python app.py
```

서버가 시작되면:
- 로컬: http://127.0.0.1:5000
- 네트워크: http://[내부IP]:5000

### 3. ngrok으로 외부 접속 (선택사항)
```bash
# ngrok 다운로드: https://ngrok.com/download
ngrok http 5000
```

## 📡 API 엔드포인트

### GET `/predict`
AI 추천 번호 생성

**응답 예시:**
```json
{
  "status": "success",
  "numbers": [3, 12, 24, 28, 35, 41],
  "message": "AI 추천 번호 생성 완료"
}
```

### GET `/`
서버 상태 확인

## 🔧 모델 재학습

새로운 데이터로 모델을 다시 학습하려면:

```bash
# 1. 최신 로또 데이터 수집 (선택사항)
python lotto_data_collector.py

# 2. 모델 학습
python train_model.py
```

## 📚 기술 스택

- **딥러닝**: PyTorch LSTM
- **서버**: Flask
- **데이터**: Pandas, NumPy
- **학습 데이터**: 제1회 ~ 최신회차 로또 당첨 번호

## ⚙️ 모델 상세

- **구조**: LSTM (2 layers, 128 hidden units)
- **입력**: 최근 20회차 당첨 번호
- **출력**: 6개 추천 번호
- **학습**: BCEWithLogitsLoss, Adam optimizer

## 📝 참고사항

- `ngrok.exe`는 30MB이므로 별도 다운로드 필요
- 모델 파일(`lotto_lstm.pth`)은 약 900KB
- 데이터는 동행복권 공식 API에서 수집

