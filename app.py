# Colab 1셀용 전체 코드: RandomForest 회귀 + RMSE 계산 + 시계열 그래프 (임의성 추가)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 샘플 데이터 생성 (또는 사용자 데이터 불러오기)
np.random.seed(42)
sample_size = 200
df = pd.DataFrame({
    'temp_100': np.random.normal(0, 1, sample_size),
    'temp_200': np.random.normal(0, 1, sample_size),
    'temp_500': np.random.normal(0, 1, sample_size),
    'press_200': np.random.normal(1000, 10, sample_size),
    'press_500': np.random.normal(500, 5, sample_size),
})
df['target'] = 0.3 * df['temp_100'] - 0.2 * df['temp_200'] + 0.5 * df['press_500'] + np.random.normal(0, 2, sample_size)

# 가상의 시간 인덱스 추가
df.index = pd.date_range(start='2024-01-01', periods=sample_size, freq='D')

# 특성과 타겟 설정
X = df[['temp_100', 'temp_200', 'temp_500', 'press_200', 'press_500']]
y = df['target']

# 학습/테스트 분리 (shuffle=False로 시계열 순서 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 임의성 추가: y_test를 약간 흔들어서 "더 현실적인" 관측값 생성
noise = np.random.normal(0, 1.5, size=len(y_test))
y_test_noisy = y_test + noise  # 관측값에 노이즈 추가

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# RMSE 및 R² 계산 (noisy한 관측값 기준)
rmse = np.sqrt(mean_squared_error(y_test_noisy, y_pred))
r2 = r2_score(y_test_noisy, y_pred)

# 결과 출력
print("RMSE (vs noisy observed):", rmse)
print("R² Score (vs noisy observed):", r2)

# 시계열 그래프 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test_noisy.values, label='Observed (noisy)', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted', marker='x')
plt.title('Observed vs Predicted Values (Time Series, with Noise)')
plt.xlabel('Date')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
