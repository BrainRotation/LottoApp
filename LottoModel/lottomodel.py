import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# --- 1. 데이터셋 클래스 정의 ---
class LottoDataset(Dataset):
    def __init__(self, data_path, window_size=20):
        self.window_size = window_size
        
        # 데이터 로드
        if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
             df = pd.read_excel(data_path)
        else:
             df = pd.read_csv(data_path)
        
        # 필요한 컬럼만 추출 (날짜, 보너스 제외하고 6개 번호만 사용)
        if 'num1' in df.columns:
            cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        else:
            cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']
            
        self.data = df[cols].values # (N, 6) 형태의 numpy array
        self.num_samples = len(self.data) - window_size
        
        print(f"데이터셋 로드 완료: 총 {len(self.data)}회차 중 학습 가능 샘플 {self.num_samples}개")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 입력(X): idx 부터 window_size 만큼의 과거 데이터
        x_raw = self.data[idx : idx + self.window_size]
        y_raw = self.data[idx + self.window_size]
        
        # X 변환: (window_size, 6) -> (window_size, 45)
        x_onehot = np.zeros((self.window_size, 45), dtype=np.float32)
        for i in range(self.window_size):
            for num in x_raw[i]:
                if 1 <= num <= 45: 
                    x_onehot[i, int(num)-1] = 1.0
        
        # y 변환: (6,) -> (45,) (멀티라벨 타겟)
        y_onehot = np.zeros(45, dtype=np.float32)
        for num in y_raw:
            if 1 <= num <= 45:
                y_onehot[int(num)-1] = 1.0
                
        return torch.FloatTensor(x_onehot), torch.FloatTensor(y_onehot)

# --- 2. LSTM 모델 정의 ---
class LottoLSTM(nn.Module):
    def __init__(self, input_size=45, hidden_size=128, num_layers=2, output_size=45):
        super(LottoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM: 과거의 흐름(Sequence)을 파악
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # Fully Connected: LSTM의 결과를 45개 번호 확률로 변환
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# --- 3. 학습 및 예측 관리 클래스 ---
class LottoDeepLearning:
    def __init__(self, data_path='data/lotto-1052.csv', window_size=20):
        self.data_path = data_path
        self.window_size = window_size
        # 호환성 문제를 피하기 위해 강제로 CPU 사용
        self.device = torch.device('cpu') 
        print(f"Device set to: {self.device}")
        self.model = LottoLSTM().to(self.device)
        
    def train(self, epochs=300, batch_size=64, lr=0.001):
        print(f"학습 시작 (Device: {self.device})")
        
        dataset = LottoDataset(self.data_path, self.window_size)
        # num_workers=0 으로 설정하여 멀티프로세싱 이슈 방지
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.5f}")
                
        print("학습 완료.")

    def save_model(self, path='lotto_lstm.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"모델 저장됨: {path}")

    def load_model(self, path='lotto_lstm.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"모델 로드됨: {path}")

    def calculate_accuracy(self, test_size=20):
        """
        최근 test_size만큼의 데이터를 떼어놓고 모델 성능을 평가함.
        """
        print(f"\n--- 모델 정확도 평가 (최근 {test_size}회차 기준) ---")
        
        dataset = LottoDataset(self.data_path, self.window_size)
        total_len = len(dataset)
        
        if total_len < test_size:
            print("데이터가 부족하여 평가를 진행할 수 없습니다.")
            return

        hit_counts = [] # 맞춘 개수 저장
        top10_hit_counts = [] # 확률 상위 10개 안에 정답이 포함된 개수
        
        self.model.eval()
        with torch.no_grad():
            for i in range(total_len - test_size, total_len):
                x_tensor, y_tensor = dataset[i]
                x_input = x_tensor.unsqueeze(0).to(self.device)
                
                out = self.model(x_input)
                probs = torch.sigmoid(out).cpu().numpy()[0] 
                
                y_numpy = y_tensor.numpy()
                real_nums = [k+1 for k, v in enumerate(y_numpy) if v == 1.0]
                
                top6_indices = probs.argsort()[-6:][::-1]
                pred_nums_top6 = [k+1 for k in top6_indices]
                
                top10_indices = probs.argsort()[-10:][::-1]
                pred_nums_top10 = [k+1 for k in top10_indices]
                
                hit = len(set(real_nums) & set(pred_nums_top6))
                hit_counts.append(hit)
                
                hit10 = len(set(real_nums) & set(pred_nums_top10))
                top10_hit_counts.append(hit10)
                
        avg_hit = sum(hit_counts) / len(hit_counts)
        avg_hit10 = sum(top10_hit_counts) / len(top10_hit_counts)
        
        print(f"평균 적중 개수 (Top 6): {avg_hit:.2f}개 / 6개")
        print(f"평균 포함 개수 (Top 10): {avg_hit10:.2f}개 / 6개 (상위 10개 추천 시)")
        print(f"최대 적중 개수: {max(hit_counts)}개")
        
        from collections import Counter
        counts = Counter(hit_counts)
        print("적중 개수별 횟수:")
        for k in sorted(counts.keys(), reverse=True):
            print(f"  {k}개 맞춤: {counts[k]}회")

    def predict(self, recent_data):
        self.model.eval()
        
        x_onehot = np.zeros((1, self.window_size, 45), dtype=np.float32)
        
        for i in range(self.window_size):
            row = recent_data[i]
            for num in row:
                if 1 <= num <= 45:
                    x_onehot[0, i, int(num)-1] = 1.0
                    
        input_tensor = torch.FloatTensor(x_onehot).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
        probs_sum = probs.sum()
        normalized_probs = probs / probs_sum
        
        recommended_numbers = np.random.choice(
            range(1, 46),
            size=6,
            replace=False,
            p=normalized_probs
        )
        
        return sorted(recommended_numbers.tolist())

# 하위 호환성을 위한 클래스 별칭 (LottoPredictor 호출 시 LottoDeepLearning 사용)
LottoPredictor = LottoDeepLearning
