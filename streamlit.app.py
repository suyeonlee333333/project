import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="무임손실 시뮬레이터", layout="wide")

# 데이터 불러오기
@st.cache_data
def load_data():
    file_path = "문제해결학습자료_혼자수정본.xlsx"
    df_train = pd.read_excel(file_path, sheet_name="학습시킬 데이터")
    df_population = pd.read_excel(file_path, sheet_name="월별 인구 수")
    
    df_train["연도-월"] = pd.to_datetime(df_train["연도-월"])
    df_population["월간 / 나이"] = pd.to_datetime(df_population["월간 / 나이"])
    df_population.rename(columns={"월간 / 나이": "연도-월"}, inplace=True)
    
    df_merged = pd.merge(df_train, df_population, on="연도-월", how="inner")
    return df_merged

# 조정 고령 인구수 계산 함수
def calculate_adjusted_elderly_fixed(df, age_threshold):
    age_cols = []
    for col in df.columns:
        try:
            col_int = int(col)
            if col_int >= age_threshold:
                age_cols.append(col)
        except:
            continue
    df['조정 고령 인구수'] = df[age_cols].sum(axis=1)
    return df

# 시뮬레이션 함수
def simulate_rf_single_variable_loss(df_original, model, age_range=(65, 75)):
    results = []
    for age in range(age_range[0], age_range[1] + 1):
        df_temp = calculate_adjusted_elderly_fixed(df_original.copy(), age_threshold=age)
        X_temp = df_temp[["조정 고령 인구수"]]
        df_temp["예측 무임인원"] = model.predict(X_temp)

        mean_loss_per_person = df_original["무임손실액 (백만)"].sum() / df_original["무임인원"].sum()
        df_temp["예측 손실액(백만)"] = df_temp["예측 무임인원"] * mean_loss_per_person

        results.append({
            "기준연령": age,
            "예측 무임인원 합계": int(df_temp["예측 무임인원"].sum()),
            "예측 손실액 합계(백만)": round(df_temp["예측 손실액(백만)"].sum(), 2)
        })

    df_result = pd.DataFrame(results)
    base_loss = df_result.loc[df_result["기준연령"] == age_range[0], "예측 손실액 합계(백만)"].values[0]
    df_result["손실액 절감액(백만)"] = df_result["예측 손실액 합계(백만)"].apply(lambda x: round(base_loss - x, 2))
    df_result["절감률(%)"] = df_result["손실액 절감액(백만)"] / base_loss * 100
    return df_result

# 모델 학습 함수
def train_model(df, age_threshold=65):
    df_adjusted = calculate_adjusted_elderly_fixed(df.copy(), age_threshold)
    X = df_adjusted[["조정 고령 인구수"]]
    y = df_adjusted["무임인원"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Main UI
st.title("🧓 고령자 무임 손실 예측 대시보드")
df_data = load_data()

# 모델 훈련
model_rf = train_model(df_data, age_threshold=65)

# 시뮬레이션
df_simulation = simulate_rf_single_variable_loss(df_data, model_rf)

# 기준연령 슬라이더
selected_age = st.slider("기준연령을 선택하세요", min_value=65, max_value=75, value=65)

# 선택된 연령 예측 결과 표시
selected_row = df_simulation[df_simulation["기준연령"] == selected_age]
st.subheader(f"🎯 기준연령 {selected_age}세의 예측 결과")
st.metric("예측 무임인원", f"{selected_row['예측 무임인원 합계'].values[0]:,} 명")
st.metric("예측 손실액", f"{selected_row['예측 손실액 합계(백만)'].values[0]:,} 백만원")
st.metric("손실액 절감액", f"{selected_row['손실액 절감액(백만)'].values[0]:,} 백만원")

# 시각화
st.subheader("📉 기준연령별 예측 손실액 및 절감률")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**기준연령 vs 예측 손실액**")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df_simulation, x="기준연령", y="예측 손실액 합계(백만)", marker="o", ax=ax1)
    ax1.set_ylabel("예측 손실액(백만)")
    st.pyplot(fig1)

with col2:
    st.markdown("**기준연령 vs 절감률(%)**")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df_simulation, x="기준연령", y="절감률(%)", palette="viridis", ax=ax2)
    ax2.set_ylabel("절감률(%)")
    st.pyplot(fig2)

# 데이터프레임 출력
st.subheader("📋 기준연령별 예측 요약 테이블")
st.dataframe(df_simulation, use_container_width=True)


