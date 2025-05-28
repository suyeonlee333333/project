import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="무임손실 시뮬레이터", layout="wide")

# 📁 데이터 불러오기
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
def calculate_adjusted_elderly_fixed(df, age_threshold):
    age_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str.isdigit() and int(col_str) >= age_threshold:
            age_cols.append(col)
    df["조정 고령 인구수"] = df[age_cols].sum(axis=1)
    return df


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

def train_model(df, age_threshold=65):
    df_adjusted = calculate_adjusted_elderly_fixed(df.copy(), age_threshold)
    X = df_adjusted[["조정 고령 인구수"]]
    y = df_adjusted["무임인원"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 🚀 실행
st.title("🧓 무임손실 예측 시뮬레이터")
st.markdown("### 👇 아래에서 분석 항목을 선택하세요")

df_data = load_data()
model_rf = train_model(df_data, age_threshold=65)
df_simulation = simulate_rf_single_variable_loss(df_data, model_rf)

# 📌 사이드바 - 기준연령 선택
with st.sidebar:
    st.header("🔧 시뮬레이션 설정")
    selected_age = st.slider("기준연령", 65, 75, 65)
    st.markdown("---")
    st.info("기준연령을 조정하면 예측 결과가 즉시 반영됩니다.")

# 📑 탭 구성
tab1, tab2, tab3 = st.tabs(["📈 요약 리포트", "📊 시각화 분석", "📋 데이터 테이블"])

# 📊 탭 1: 요약 리포트
with tab1:
    st.subheader(f"🎯 기준연령 {selected_age}세의 예측 리포트")
    selected_row = df_simulation[df_simulation["기준연령"] == selected_age]
    col1, col2, col3 = st.columns(3)
    col1.metric("예측 무임인원", f"{selected_row['예측 무임인원 합계'].values[0]:,} 명")
    col2.metric("예측 손실액", f"{selected_row['예측 손실액 합계(백만)'].values[0]:,} 백만원")
    col3.metric("손실액 절감액", f"{selected_row['손실액 절감액(백만)'].values[0]:,} 백만원")
    
    with st.expander("📌 해석 가이드"):
        st.markdown("""
        - **예측 무임인원**: 기준연령을 기준으로 예상되는 전체 무임 승차 인원입니다.  
        - **예측 손실액**: 무임인원에 따라 추정된 손실액 (백만원 단위)입니다.  
        - **손실액 절감액**: 기준연령을 올렸을 때 절감 가능한 비용입니다.
        """)

# 📊 탭 2: 시각화
with tab2:
    st.subheader("📉 기준연령별 예측 손실액 변화")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df_simulation, x="기준연령", y="예측 손실액 합계(백만)", marker="o", ax=ax1)
    ax1.set_ylabel("예측 손실액(백만)")
    st.pyplot(fig1)

    st.subheader("🟩 기준연령별 손실 절감률")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df_simulation, x="기준연령", y="절감률(%)", palette="crest", ax=ax2)
    ax2.set_ylabel("절감률(%)")
    st.pyplot(fig2)

# 📋 탭 3: 데이터 테이블
with tab3:
    st.subheader("📋 기준연령별 예측 요약 테이블")
    st.dataframe(df_simulation, use_container_width=True)
