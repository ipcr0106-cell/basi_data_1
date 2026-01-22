import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly
import platform

baci_hs=pd.read_csv("baci_85_only.csv")
baci_country=pd.read_csv("country_codes_V202501.csv")
print(baci_hs.head())
print(baci_hs.info())
print(baci_hs.columns)
print(baci_country.head())
print(baci_country.info())
print(baci_country.columns)

print(len(baci_hs))

# i: 수출 국가는 한국(410)만 남기고 다 삭제
# 한국이 가진 여러 수입국가
# 남은 데이터의 연도를 바꿔 변화 추이 관찰

# i에서 한국 빼고 삭제
    # baci_korea=baci_hs[baci_hs["i"]==410].copy()
    # print(len(baci_korea))
    # print(baci_korea.head())
    # baci_korea.to_csv("baci_korea_only.csv",index=False) # 새로운 파일로 저장 -> 원본 파일 삭제하고 주석으로 넘기기

# k에서 반도체에서 'smart card'만 빼고 삭제
    # baci_85=baci_hs[baci_hs["k"]==852352].copy()
    # print(len(baci_85))
    # print(baci_85)
    # baci_85.to_csv("baci_85_only.csv",index=False) # 새로운 파일로 저장 -> 원본 파일(baci_korea_only) 삭제하고 주석으로 넘기기(나중에 다른 상품에 대해 하느 싶으면 남겨놓기)


# j를 국가이름으로 바꾸기(merge, mapping)
baci_country=baci_country.rename(columns={'country_code':"j"}) # country code 컬럼 명을 j로 바꿔주기
baci_final=pd.merge(baci_hs,baci_country,on="j",how="left") # j열로 병합, 왼쪽 기준
print(baci_final)
# i를 대한민국으로 바꾸기

# t(연도)를 랜덤하게 바꾸기
years = [2021, 2022, 2023]
baci_final['t'] = np.random.choice(years, size=len(baci_final))

# 3. 결과 확인
print(baci_final['t'].value_counts()) # 각 연도별로 데이터가 잘 분산되었는지 확인
print(baci_final.head())

# 4. 변경된 데이터 저장 (필요 시)
# df.to_csv('baci_85_randomized.csv', index=False)


print("------------------------------------------------------")

# 1. 한글 깨짐 방지 설정 (OS별 자동 설정)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else: # Linux (Colab 등)
    plt.rcParams['font.family'] = 'NanumGothic'
print("------------------------------------------------------")

# 연도별 total value(금액) 막대그래프로 그리기
yearly_total = baci_final.groupby('t')['v'].sum()

# 2. 막대그래프(Bar Chart) 시각화
plt.figure(figsize=(10, 6))
yearly_total.plot(kind='bar', color='skyblue', edgecolor='black')

# 3. 그래프 디테일 설정 (CTO 수준의 깔끔한 리포트용)
plt.title('연도별 총 수출액 추이 (2021-2023)', fontsize=15, pad=15)
plt.xlabel('연도 (Year)', fontsize=12)
plt.ylabel('총 수출액 (Unit: 1,000 USD)', fontsize=12)
plt.xticks(rotation=0)  # 연도 라벨을 가로로 표시
plt.grid(axis='y', linestyle='--', alpha=0.7) # 가로 점선 추가로 가독성 향상

# 4. 수치 표시 (막대 위에 금액 텍스트 추가)
for i, v in enumerate(yearly_total):
    plt.text(i, v + (v * 0.01), f"{v:,.0f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


print("------------------------------------------------------")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches # patches 임포트 필수!
import numpy as np
import platform

# 1. 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 트리맵 레이아웃 알고리즘 (가중치 기반 분할)
def treemap_layout(values, x, y, w, h):
    if len(values) == 0: return []
    if len(values) == 1: return [(x, y, w, h)]
    
    # [수정] 단순히 반으로 나누는 것이 아니라, 합계(Value)가 비슷한 지점을 찾아 분할
    total_sum = sum(values)
    acc = 0
    split_idx = 1
    for i, v in enumerate(values):
        acc += v
        if acc >= total_sum / 2:
            split_idx = i + 1
            break
    if split_idx >= len(values): split_idx = len(values) - 1
    
    v1, v2 = values[:split_idx], values[split_idx:]
    s1 = sum(v1)
    
    rects = []
    if w > h: # 가로 분할
        w1 = w * (s1 / total_sum)
        rects.extend(treemap_layout(v1, x, y, w1, h))
        rects.extend(treemap_layout(v2, x + w1, y, w - w1, h))
    else: # 세로 분할
        h1 = h * (s1 / total_sum)
        rects.extend(treemap_layout(v1, x, y, w, h1))
        rects.extend(treemap_layout(v2, x, y + h1, w, h - h1))
    return rects

# 3. 데이터 준비 (상위 15개국으로 제한해야 '통'으로 안 보입니다)
# baci_final 데이터에서 상위 15개만 추출
top_data = baci_final.groupby('country_name')['v'].sum().sort_values(ascending=False).head(15)
values = top_data.values.tolist()
labels = top_data.index.tolist()

# 4. 시각화 실행
fig, ax = plt.subplots(figsize=(14, 10))
rects = treemap_layout(values, 0, 0, 100, 100) # 100x100 영역 안에서 분할

# [수정] cmap은 배열이므로 인덱싱 [] 사용
cmap = plt.cm.Spectral(np.linspace(0, 1, len(values)))

for i, (rect, label, val) in enumerate(zip(rects, labels, values)):
    x, y, w, h = rect
    # patches.Rectangle을 사용하여 사각형 추가
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='white', facecolor=cmap[i], alpha=0.8))
    
    # 사각형 면적이 일정 크기 이상일 때만 텍스트 표시
    if w > 4 and h > 4:
        plt.text(x + w/2, y + h/2, f"{label}\n{val:,.0f}", ha='center', va='center', fontsize=10, fontweight='bold')

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('off')
plt.title('상위 15개국 수출 규모 비중 트리맵', fontsize=18, pad=20)
plt.tight_layout()
plt.show()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import platform

# 1. 스트림릿 페이지 설정
st.set_page_config(page_title="2025 무역 데이터 분석 CTO 대시보드", layout="wide")

# 2. 한글 폰트 설정 함수 (환경 대응)
def set_korean_font():
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 3. 데이터 로드 및 전처리 (캐싱 처리로 속도 향상)
@st.cache_data
def get_processed_data():
    # 파일 로드
    baci_hs = pd.read_csv("baci_85_only.csv")
    baci_country = pd.read_csv("country_codes_V202501.csv")
    
    # j를 국가이름으로 바꾸기
    baci_country = baci_country.rename(columns={'country_code': "j"})
    baci_final = pd.merge(baci_hs, baci_country, on="j", how="left")
    
    # t(연도) 랜덤 생성 (요청 사항)
    np.random.seed(42)
    years = [2021, 2022, 2023]
    baci_final['t'] = np.random.choice(years, size=len(baci_final))
    
    return baci_final

df = get_processed_data()

# 4. 트리맵 레이아웃 알고리즘
def treemap_layout(values, x, y, w, h):
    if len(values) == 0: return []
    if len(values) == 1: return [(x, y, w, h)]
    
    total_sum = sum(values)
    acc = 0
    split_idx = 1
    for i, v in enumerate(values):
        acc += v
        if acc >= total_sum / 2:
            split_idx = i + 1
            break
    if split_idx >= len(values): split_idx = len(values) - 1
    
    v1, v2 = values[:split_idx], values[split_idx:]
    s1 = sum(v1)
    
    rects = []
    if w > h:
        w1 = w * (s1 / total_sum)
        rects.extend(treemap_layout(v1, x, y, w1, h))
        rects.extend(treemap_layout(v2, x + w1, y, w - w1, h))
    else:
        h1 = h * (s1 / total_sum)
        rects.extend(treemap_layout(v1, x, y, w, h1))
        rects.extend(treemap_layout(v2, x, y + h1, w, h - h1))
    return rects

# --- 사이드바: 필터링 컨트롤 ---
st.sidebar.header("📊 분석 필터 설정")
selected_years = st.sidebar.multiselect("분석 연도 선택", options=[2021, 2022, 2023], default=[2021, 2022, 2023])
top_n = st.sidebar.slider("트리맵 표시 국가 수", 5, 20, 15)

# 필터링 적용
filtered_df = df[df['t'].isin(selected_years)]

# --- 메인 대시보드 레이아웃 ---
st.title("📈 2025 글로벌 무역 데이터 분석 대시보드")
st.markdown(f"**대상 상품:** Smart Card (HS 8523.52) | **분석 국가:** 대한민국(410) 기준")

col1, col2 = st.columns(2)

# 좌측: 연도별 총 수출액 추이
with col1:
    st.subheader("🗓️ 연도별 총 수출액 추이")
    yearly_total = filtered_df.groupby('t')['v'].sum()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    yearly_total.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('연도별 수출액 합계 (1,000 USD)', fontsize=12)
    ax.set_xlabel('연도', fontsize=10)
    plt.xticks(rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(yearly_total):
        ax.text(i, v + (v * 0.01), f"{v:,.0f}", ha='center', fontweight='bold')
    
    st.pyplot(fig)

# 우측: 트리맵 비중 분석
with col2:
    st.subheader(f"🌍 상위 {top_n}개국 수출 비중 (면적 기반)")
    top_data = filtered_df.groupby('country_name')['v'].sum().sort_values(ascending=False).head(top_n)
    values = top_data.values.tolist()
    labels = top_data.index.tolist()

    if values:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        rects = treemap_layout(values, 0, 0, 100, 100)
        cmap = plt.cm.Spectral(np.linspace(0, 1, len(values)))

        for i, (rect, label, val) in enumerate(zip(rects, labels, values)):
            rx, ry, rw, rh = rect
            ax2.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='white', facecolor=cmap[i], alpha=0.8))
            if rw > 4 and rh > 4:
                ax2.text(rx + rw/2, ry + rh/2, f"{label}\n{val:,.0f}", ha='center', va='center', fontsize=9, fontweight='bold')

        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.axis('off')
        st.pyplot(fig2)
    else:
        st.warning("선택된 데이터가 없습니다.")

# 하단: 데이터 상세 정보
with st.expander("📄 데이터 상세 보기"):
    st.dataframe(filtered_df, use_container_width=True)