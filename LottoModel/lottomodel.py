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
        
    def train(self, epochs=1000, batch_size=64, lr=0.001, patience=50):
        print(f"학습 시작 (Device: {self.device})")
        print(f"설정: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={lr}")
        print(f"Early Stopping: {patience}번 연속 개선 없으면 조기 종료\n")
        
        dataset = LottoDataset(self.data_path, self.window_size)
        
        # 학습/검증 데이터 분리 (80:20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Early Stopping 변수
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"학습 데이터: {train_size}개, 검증 데이터: {val_size}개\n")
        print("="*80)
        
        for epoch in range(epochs):
            # === 학습 단계 ===
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 정확도 계산 (상위 6개 예측 중 실제 번호와 일치하는 개수)
                probs = torch.sigmoid(outputs)
                pred_indices = torch.topk(probs, k=6, dim=1).indices
                
                for i in range(len(batch_y)):
                    true_nums = torch.where(batch_y[i] > 0.5)[0]
                    pred_nums = pred_indices[i]
                    matches = len(set(true_nums.tolist()) & set(pred_nums.tolist()))
                    train_correct += matches
                    train_total += 6
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = (train_correct / train_total) * 100
            
            # === 검증 단계 ===
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # 검증 정확도 계산
                    probs = torch.sigmoid(outputs)
                    pred_indices = torch.topk(probs, k=6, dim=1).indices
                    
                    for i in range(len(batch_y)):
                        true_nums = torch.where(batch_y[i] > 0.5)[0]
                        pred_nums = pred_indices[i]
                        matches = len(set(true_nums.tolist()) & set(pred_nums.tolist()))
                        val_correct += matches
                        val_total += 6
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = (val_correct / val_total) * 100
            
            # 진행 상황 출력 (10 에포크마다)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:4d}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.5f} (정확도: {train_accuracy:.2f}%) | "
                      f"Val Loss: {avg_val_loss:.5f} (정확도: {val_accuracy:.2f}%)")
            
            # Early Stopping 체크
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  ✅ 검증 손실 개선! (Best: {best_val_loss:.5f})")
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n{'='*80}")
                    print(f"⚠️  Early Stopping 발동! ({patience}번 연속 개선 없음)")
                    print(f"   최종 에포크: {epoch+1}/{epochs}")
                    print(f"   최상 검증 손실: {best_val_loss:.5f}")
                    print(f"{'='*80}\n")
                    
                    # 최상의 모델로 복원
                    self.model.load_state_dict(best_model_state)
                    break
        
        # 학습 종료
        if patience_counter < patience:
            print(f"\n{'='*80}")
            print(f"✅ 학습 완료! (전체 {epochs} 에포크)")
            print(f"   최종 검증 손실: {avg_val_loss:.5f}")
            print(f"   최종 검증 정확도: {val_accuracy:.2f}%")
            print(f"{'='*80}\n")
            
            # 최상의 모델로 복원
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
        
        print("모델이 최상의 성능 상태로 설정되었습니다.")

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

    def _validate_combination(self, nums, prev_round=None):
        """
        로또 번호 조합이 통계적 규칙에 맞는지 검증
        점수가 높을수록 좋은 조합
        """
        score = 100  # 기본 점수
        nums_sorted = sorted(nums)
        
        # 규칙 1: 십의 자리 분포 (0-1-1-1-0 패턴에 가까울수록 좋음)
        tens_dist = [0] * 5
        for num in nums:
            tens_dist[num // 10] += 1
        
        # 이상적인 분포: [1, 1, 1, 1, 0~1] 정도
        if tens_dist[0] <= 2 and tens_dist[1] <= 2 and tens_dist[2] <= 2 and tens_dist[3] <= 2 and tens_dist[4] <= 1:
            score += 20
        else:
            score -= 10
            
        # 규칙 2: 끝자리 종류 (4-5개가 이상적)
        last_digits = len(set([n % 10 for n in nums]))
        if 4 <= last_digits <= 5:
            score += 15
        elif last_digits == 3 or last_digits == 6:
            score += 5
        
        # 규칙 3: 구간별 분포 (1-15, 16-30, 31-45 각각 1-3개)
        sections = [0, 0, 0]
        for num in nums:
            if 1 <= num <= 15:
                sections[0] += 1
            elif 16 <= num <= 30:
                sections[1] += 1
            else:
                sections[2] += 1
        
        if all(1 <= s <= 3 for s in sections):
            score += 15
        
        # 규칙 4: 연속번호 쌍 (0-1개가 일반적)
        consecutive = 0
        for i in range(len(nums_sorted)-1):
            if nums_sorted[i+1] - nums_sorted[i] == 1:
                consecutive += 1
        
        if consecutive <= 1:
            score += 10
        elif consecutive >= 3:
            score -= 15
        
        # 규칙 5: 전회차 번호 재출현 (40-50% 확률로 1개)
        if prev_round is not None:
            prev_set = set(prev_round)
            matches = len(prev_set & set(nums))
            if matches == 1:
                score += 20  # 1개 일치 (가장 흔한 패턴)
            elif matches == 0:
                score += 10  # 0개도 흔함
            elif matches == 2:
                score += 5   # 2개는 덜 흔함
            else:
                score -= 10  # 3개 이상은 드묾
        
        # 규칙 6: 동숙번호 쌍 회피 (12/21, 13/31, 14/41, 23/32, 24/42, 34/43)
        pair_rules = [(12, 21), (13, 31), (14, 41), (23, 32), (24, 42), (34, 43)]
        for p1, p2 in pair_rules:
            if p1 in nums and p2 in nums:
                score -= 20  # 동숙번호는 피하기
                break
        
        # 규칙 7: 합계 범위 (너무 작거나 크지 않게)
        total = sum(nums)
        if 100 <= total <= 200:  # 평균적인 범위
            score += 10
        
        return score

    def predict(self, recent_data):
        """
        LSTM 모델 + 통계 규칙 기반 예측 (다양성 강화)
        매번 다른 조합을 생성하기 위해 랜덤성 추가
        """
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
        
        # 전회차 번호 가져오기
        prev_round = recent_data[-1] if len(recent_data) > 0 else None
        
        # 다양성 추가: 확률에 약간의 랜덤 노이즈 추가
        noise = np.random.normal(0, 0.05, probs.shape)  # 5% 노이즈
        probs_with_noise = np.clip(probs + noise, 0, 1)
        
        # 상위 확률 번호들 선택 (매번 조금씩 다름)
        top_count = np.random.randint(18, 25)  # 18~24개 사이에서 랜덤
        top_numbers = np.argsort(probs_with_noise)[-top_count:][::-1] + 1
        
        candidates_with_scores = []
        
        # 더 많은 조합 생성 및 평가 (100개)
        for _ in range(100):
            # 상위 번호들의 확률 기반 샘플링
            top_probs = probs_with_noise[top_numbers - 1]
            top_probs_norm = top_probs / top_probs.sum()
            
            candidate = np.random.choice(
                top_numbers,
                size=6,
                replace=False,
                p=top_probs_norm
            )
            
            # 규칙 기반 점수 계산
            score = self._validate_combination(candidate, prev_round)
            
            # 점수가 일정 수준 이상이면 후보에 추가
            if score >= 80:  # 80점 이상만 허용
                candidates_with_scores.append((score, candidate))
        
        # 후보가 없으면 점수 낮춰서 재시도
        if not candidates_with_scores:
            for _ in range(50):
                top_probs = probs_with_noise[top_numbers - 1]
                top_probs_norm = top_probs / top_probs.sum()
                
                candidate = np.random.choice(
                    top_numbers,
                    size=6,
                    replace=False,
                    p=top_probs_norm
                )
                
                score = self._validate_combination(candidate, prev_round)
                if score >= 60:  # 60점 이상
                    candidates_with_scores.append((score, candidate))
        
        # 상위 점수 조합 중에서 랜덤하게 선택 (다양성!)
        if candidates_with_scores:
            # 점수 순으로 정렬
            candidates_with_scores.sort(reverse=True, key=lambda x: x[0])
            
            # 상위 30% 중에서 랜덤 선택
            top_30_percent = max(1, len(candidates_with_scores) // 3)
            selected_idx = np.random.randint(0, min(top_30_percent, len(candidates_with_scores)))
            
            _, selected_combination = candidates_with_scores[selected_idx]
            return sorted(selected_combination.tolist())
        else:
            # 최악의 경우: 순수 확률 기반 선택
            return sorted(np.random.choice(
                range(1, 46),
                size=6,
                replace=False,
                p=probs_with_noise / probs_with_noise.sum()
            ).tolist())

# 하위 호환성을 위한 클래스 별칭 (LottoPredictor 호출 시 LottoDeepLearning 사용)
LottoPredictor = LottoDeepLearning
