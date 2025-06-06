# **선택 장애 실험: 선택지가 많을수록 의사결정이 어려워질까?**
## **📌 개요**  
선택지가 많을수록 사람들이 더 좋은 결정을 내릴까, 아니면 오히려 혼란스러워질까?  
이 실험은 **선택의 역설(The Paradox of Choice)** 개념을 검증하며, 선택지 개수가 의사결정 속도와 만족도에 미치는 영향을 분석한다.

---

## **🛠️ 실험 설계**
### **1️⃣ 실험 참가자 모집**
- 최소 50명 이상의 실험 참가자를 모집 (더 많을수록 신뢰도↑)
- 참가자들은 **두 그룹(A/B) 중 하나**에 랜덤으로 배정됨

### **2️⃣ 선택 조건 설정**
- **그룹 A**: **3가지 선택지** 제공  
- **그룹 B**: **20가지 선택지** 제공  
- 두 그룹은 동일한 유형의 선택을 해야 함 (예: 아이스크림 맛 선택, 티셔츠 디자인 선택 등)  

### **3️⃣ 측정할 변수**
| 변수 | 설명 | 측정 방법 |
|------|------|------|
| **의사결정 시간** | 선택하는 데 걸린 시간 | 초 단위로 기록 |
| **선택 만족도** | 선택 후 만족도 평가 | 1~10점 척도로 설문 |
| **선택 후 후회 정도** | 다시 선택할 기회가 있다면 바꿀 것인지 | "예/아니오" 응답 |

### **4️⃣ 실험 과정**
1. 참가자는 무작위로 **그룹 A 또는 그룹 B**에 배정됨  
2. 주어진 선택지에서 원하는 항목을 선택 (예: "어떤 아이스크림 맛을 고를래?")  
3. **선택에 걸린 시간**을 초 단위로 기록  
4. 선택 후 **만족도와 후회 정도를 설문 조사**  
5. 결과 데이터를 수집하고 비교 분석  

---

## **📊 데이터 분석 방법**
### **1️⃣ 평균 비교 (A vs B)**
- 그룹 A(3가지 선택)와 그룹 B(20가지 선택)의 **의사결정 시간 비교**  
- 평균 만족도 점수 차이 분석  

### **2️⃣ 상관관계 분석**
- 선택지 개수 vs **의사결정 속도**의 상관관계  
- 선택지 개수 vs **만족도**의 관계  

### **3️⃣ 분산 분석 (ANOVA)**
- 선택지 개수가 의사결정 시간과 만족도에 유의미한 영향을 주는지 검증  

---

## **📐 수학적 모델링**
이 실험을 수학적으로 모델링하면, **의사결정 시간(T)**은 선택지 개수(N)에 따라 다음과 같이 정의될 수 있음.

### **1️⃣ 로그 모델 (Logarithmic Model)**
의사결정 시간이 **선택지 개수의 로그 함수**로 증가한다고 가정  
\[
T(N) = a \log(N) + b
\]
- 여기서 \(a, b\)는 경험적으로 결정되는 상수  
- 선택지가 많아질수록 시간이 증가하지만 **완만한 증가 곡선**을 보임  

### **2️⃣ 선형 모델 (Linear Model)**
단순하게 선택지 개수가 많을수록 **의사결정 시간이 선형적으로 증가**한다고 가정  
\[
T(N) = aN + b
\]
- 여기서 \(a, b\)는 경험적으로 측정되는 상수  
- 선택지가 많을수록 의사결정 시간이 계속 증가  

### **3️⃣ 만족도의 역U자형 모델 (Inverted-U Model)**
이전 연구에 따르면, 선택지가 너무 적거나 너무 많으면 **만족도가 낮아지고, 중간일 때 가장 높음**  
\[
S(N) = c - d(N - N_{opt})^2
\]
- \(N_{opt}\) : 최적의 선택지 개수  
- \(c, d\) : 실험 데이터로 결정되는 상수  
- 만족도 곡선이 **역U자형 (∩ 형태)**이 되며, 특정 개수에서 최대 만족도 도달  

---

## **🔍 기대 결과**
| 가설 | 예상 결과 |
|------|------|
| **선택지가 많으면 의사결정 시간이 길어진다** | 그룹 B(20개 선택지)가 그룹 A(3개 선택지)보다 더 오래 걸릴 것 |
| **선택지가 많으면 만족도가 낮아질 수 있다** | 그룹 B가 선택 후 후회할 가능성이 더 높을 것 |
| **최적의 선택지 개수 존재** | 만족도가 최대가 되는 특정 개수(예: 5~7개)가 있을 것 |

---

## **📌 프로젝트 적용 가능성**
✅ **UX/UI 디자인 연구** → 웹사이트나 앱에서 최적의 선택 개수 설계  
✅ **마케팅 및 상품 진열 전략** → 소비자들이 적절한 선택을 할 수 있도록 최적의 상품 개수 설정  
✅ **인지심리학 연구** → 선택 과부하(Choice Overload) 현상 검증  
