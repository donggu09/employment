# streamlit_app.py
"""
Streamlit 대시보드 (한국어)
- 목적: '기후 변화가 취업에 미치는 영향' 주제로
  1) 공식 공개 데이터 대시보드 (NASA GISTEMP, World Bank 등 시도)
  2) 사용자 입력(프롬프트 텍스트 기반) 대시보드 (프롬프트에 제공된 내용만 사용)
- 구현 규칙:
  - 데이터 표준화: date, value, group(optional)
  - 전처리: 결측 처리 / 형변환 / 중복 제거 / 미래 데이터 제거
  - 캐싱: @st.cache_data 사용
  - CSV 다운로드 버튼 제공
  - 모든 UI는 한국어
- 출처(예시):
  * NASA GISTEMP table CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
  * World Bank API (Employment in industry %): http://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS?format=json&per_page=20000
  * (앱 실행 환경에서 API 호출이 실패하면 예시 데이터로 자동 대체)
"""

import os
import io
import time
import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests

# plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 설정
# ---------------------------
st.set_page_config(page_title="기후와 취업: 데이터 대시보드", layout="wide")
TODAY = datetime.datetime.now().date()  # "오늘" 기준, 로컬 환경 시간 사용

# ---------------------------
# 폰트(Pretendard) 적용 시도
# ---------------------------
def try_apply_pretendard():
    font_path = "/fonts/Pretendard-Bold.ttf"
    if os.path.exists(font_path):
        # Streamlit (CSS)
        css = f"""
        <style>
        @font-face {{
            font-family: 'PretendardCustom';
            src: url('{font_path}') format('truetype');
            font-weight: 700;
            font-style: normal;
        }}
        html, body, [class*="css"] {{
            font-family: 'PretendardCustom', Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        # Matplotlib
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'PretendardCustom'
        except Exception:
            pass

try_apply_pretendard()

# ---------------------------
# 유틸: HTTP 재시도
# ---------------------------
def retry_get(url: str, params: dict = None, headers: dict = None, timeout: int = 15, max_retries: int = 2, backoff: float = 1.0) -> Optional[requests.Response]:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception:
            if attempt < max_retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return None

# ---------------------------
# 공개 데이터 로드 함수 (캐시)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_gistemp_csv():
    """
    NASA GISTEMP (global monthly anomalies)
    출처 주석: https://data.giss.nasa.gov/gistemp/
    CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    resp = retry_get(url, max_retries=2)
    if resp is None:
        return None
    content = resp.content.decode('utf-8', errors='replace')
    try:
        # 파일은 첫 행이 header(Year, Jan...Dec, J-D, D-N, DJF, etc.)
        # skiprows=1 to skip an initial comment line if present
        df = pd.read_csv(io.StringIO(content), skiprows=1)
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        # Keep months Jan-Dec
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        # If months present in columns -> melt
        present_months = [m for m in months if m in df.columns]
        if not present_months:
            return None
        df_long = df.melt(id_vars=['Year'], value_vars=present_months, var_name='Month', value_name='Anomaly')
        # build date from Year and Month
        month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
        def make_date(row):
            try:
                y = int(row['Year'])
                m = month_map.get(row['Month'])
                return datetime.date(y, m, 1)
            except Exception:
                return None
        df_long['date'] = df_long.apply(make_date, axis=1)
        df_long = df_long[df_long['date'].notnull()]
        df_long['value'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')
        df_long = df_long[['date','value']]
        # remove future dates
        df_long = df_long[df_long['date'].apply(lambda d: d <= TODAY)]
        df_long = df_long.dropna(subset=['value']).sort_values('date').reset_index(drop=True)
        df_long['group'] = '지구 평균 온도 이상치(℃)'
        return df_long
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment():
    """
    World Bank API: Employment in industry (% of total employment) - SL.IND.EMPL.ZS
    API: http://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS?format=json&per_page=20000
    출처 주석: https://data.worldbank.org/indicator/SL.IND.EMPL.ZS
    """
    api = "http://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS"
    params = {'format':'json','per_page':20000}
    resp = retry_get(api, params=params, max_retries=2)
    if resp is None:
        return None
    try:
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        records = data[1]
        rows = []
        for rec in records:
            country = rec.get('country', {}).get('value')
            year = rec.get('date')
            val = rec.get('value')
            if country is None or year is None or val is None:
                continue
            try:
                y = int(year)
            except Exception:
                continue
            # Use Jan 1 of the year as date
            d = datetime.date(y, 1, 1)
            # Filter future years here (we'll still re-filter later)
            if d > TODAY:
                continue
            rows.append({'date':d, 'group': country, 'value': float(val)})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return None

# ---------------------------
# 예시(대체) 데이터 (공개 데이터 호출 실패 시 사용)
# ---------------------------
def sample_climate_data():
    years = list(range(TODAY.year - 14, TODAY.year + 1))
    dates = [datetime.date(y,1,1) for y in years]
    values = np.round(np.linspace(0.4, 1.1, len(years)) + np.random.normal(0, 0.05, len(years)), 3)
    df = pd.DataFrame({'date': dates, 'value': values})
    df['group'] = '지구 평균 온도 이상치(℃)'
    return df

def sample_employment_data():
    years = list(range(TODAY.year - 9, TODAY.year + 1))
    rows = []
    for y in years:
        rows.append({'date': datetime.date(y,1,1), 'group':'한국(예시)', 'value': float(24.0 + np.random.normal(0,0.8))})
        rows.append({'date': datetime.date(y,1,1), 'group':'OECD 평균(예시)', 'value': float(22.0 + np.random.normal(0,0.8))})
    return pd.DataFrame(rows)

# ---------------------------
# 전처리 함수 (캐시)
# ---------------------------
@st.cache_data
def remove_future_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy[df_copy[date_col].notnull()]
    df_copy = df_copy[df_copy[date_col].dt.date <= TODAY]
    return df_copy

@st.cache_data
def preprocess_public_climate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d = d.sort_values('date').drop_duplicates(subset=['date'])
    d = d[d['date'].dt.date <= TODAY]
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    if d['value'].isnull().any():
        d['value'] = d['value'].interpolate().fillna(method='bfill').fillna(method='ffill')
    d = d.dropna(subset=['value']).reset_index(drop=True)
    return d

@st.cache_data
def preprocess_public_employment(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.drop_duplicates(subset=['date','group']) if {'date','group'}.issubset(d.columns) else d.drop_duplicates()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d = d[d['date'].dt.date <= TODAY]
    # ensure value numeric
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    # sort
    if 'group' in d.columns:
        d = d.sort_values(['group','date']).reset_index(drop=True)
        # Use transform to avoid index-mismatch errors
        d['value'] = d.groupby('group')['value'].transform(lambda s: s.interpolate().fillna(method='bfill').fillna(method='ffill'))
    else:
        d = d.sort_values('date').reset_index(drop=True)
        d['value'] = d['value'].interpolate().fillna(method='bfill').fillna(method='ffill')
    d = d.dropna(subset=['value']).reset_index(drop=True)
    return d

# ---------------------------
# 공개 데이터 불러오기 (시도 -> 재시도 -> 대체)
# ---------------------------
st.sidebar.title("데이터 로드 상태")
with st.spinner("공식 공개 데이터 불러오는 중..."):
    climate_raw = fetch_gistemp_csv()
    emp_raw = fetch_worldbank_employment()

# simple retry if None
if climate_raw is None:
    climate_raw = fetch_gistemp_csv()
if emp_raw is None:
    emp_raw = fetch_worldbank_employment()

# fallback
if climate_raw is None:
    st.sidebar.error("NASA GISTEMP 데이터 로드 실패 — 예시 데이터 사용")
    climate_raw = sample_climate_data()
else:
    st.sidebar.success("NASA GISTEMP 데이터 로드 성공")

if emp_raw is None:
    st.sidebar.error("World Bank 고용 데이터 로드 실패 — 예시 데이터 사용")
    emp_raw = sample_employment_data()
else:
    st.sidebar.success("World Bank 고용 데이터 로드 성공 (원본 불러옴)")

# 전처리
climate_df = preprocess_public_climate(climate_raw)
employment_df = preprocess_public_employment(emp_raw)

# ---------------------------
# UI: 탭 구조
# ---------------------------
tab1, tab2 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력(리포트 기반) 대시보드"])

with tab1:
    st.header("공식 공개 데이터 기반 분석")
    st.markdown("**출처(예시)**: NASA GISTEMP (기후 이상치), World Bank (산업별 고용 비율).")
    col_left, col_right = st.columns([2,1])

    with col_left:
        st.subheader("지구 평균 온도 이상치 (시간 흐름)")
        if not climate_df.empty:
            fig = px.line(climate_df, x='date', y='value', title='지구 평균 온도 이상치 (월별/연별)', labels={'date':'날짜','value':'이상치 (℃)'}, hover_data={'date':True,'value':True})
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("기후(전처리) CSV 다운로드", climate_df.to_csv(index=False).encode('utf-8-sig'), "climate_preprocessed.csv", "text/csv")
        else:
            st.warning("기후 데이터가 없습니다.")

    with col_right:
        st.subheader("산업별 고용 비율 (요약)")
        if not employment_df.empty:
            # Show latest year summary (median per country)
            try:
                employment_df['year'] = employment_df['date'].dt.year
                latest_year = int(employment_df['year'].max())
                latest = employment_df[employment_df['year'] == latest_year]
                display_table = latest.sort_values('value', ascending=False).head(10)[['group','value']].rename(columns={'group':'국가/그룹','value':'산업 고용 비율(%)'})
                st.table(display_table.reset_index(drop=True))
            except Exception:
                st.write("고용 데이터 요약을 생성하지 못했습니다.")
            st.download_button("고용(전처리) CSV 다운로드", employment_df.to_csv(index=False).encode('utf-8-sig'), "employment_preprocessed.csv", "text/csv")
        else:
            st.warning("고용 데이터가 없습니다.")

    st.markdown("---")
    st.subheader("기후(온도 이상치) vs 산업 고용(연 단위 비교)")
    # 연 단위 집계 후 비교 (안 겹치면 안내)
    try:
        c_ann = climate_df.copy()
        c_ann['year'] = c_ann['date'].dt.year
        c_ann_agg = c_ann.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})

        e_ann = employment_df.copy()
        e_ann['year'] = e_ann['date'].dt.year
        e_ann_agg = e_ann.groupby('year')['value'].median().reset_index().rename(columns={'value':'industry_employment_median'})

        merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')
        if merged.empty:
            st.info("공개 데이터 간 공통 연도가 적어 비교 그래프를 만들 수 없습니다. (예시 데이터일 수 있음)")
        else:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Line(x=merged['year'], y=merged['temp_anomaly'], name='기후 이상치(연평균)', yaxis='y1'))
            fig_comp.add_trace(go.Line(x=merged['year'], y=merged['industry_employment_median'], name='산업 고용(연도별 중앙값)', yaxis='y2'))
            fig_comp.update_layout(
                title="연도별: 기후 이상치 vs 산업 고용(중앙값)",
                xaxis_title="연도",
                yaxis=dict(title="기후 이상치(℃)"),
                yaxis2=dict(title="산업 고용(%)", overlaying='y', side='right')
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    except Exception:
        st.write("비교 그래프 생성 중 오류가 발생했습니다.")

    st.markdown("**참고**: API 호출이 실패하면 예시 데이터로 자동 대체됩니다. (사이드바에 상태 표시)")

with tab2:
    st.header("사용자 입력(프롬프트 기반) 대시보드")
    st.markdown("입력: 사용자가 제공한 보고서 텍스트(프롬프트)만을 바탕으로 생성한 예시 데이터입니다. 앱 실행 중 파일 업로드나 추가 텍스트 입력을 요구하지 않습니다.")

    # --- 사용자 데이터 생성 (프롬프트 기반, 제공된 리포트 내용만 사용) ---
    # 1) 녹색 일자리 증가 vs 화석일자리 감소 (연도별 지표, 2018-2024)
    years = list(range(2018, min(TODAY.year, 2025)+1))
    if not years:
        years = [TODAY.year]
    green = [5,8,12,15,18,22,26][:len(years)]
    fossil = [0,-2,-4,-6,-9,-12,-15][:len(years)]
    user_jobs_df = pd.DataFrame({
        'date': [datetime.date(y,1,1) for y in years],
        'group': ['녹색 일자리 증가'] * len(years),
        'value': green
    })
    user_jobs_df = pd.concat([user_jobs_df,
                              pd.DataFrame({'date':[datetime.date(y,1,1) for y in years],
                                            'group':['화석연료 기반 일자리 감소']*len(years),
                                            'value': fossil})],
                             ignore_index=True)
    user_jobs_df = remove_future_dates(user_jobs_df, 'date')
    user_jobs_df['date'] = pd.to_datetime(user_jobs_df['date'])

    # 2) 전공별 취업률 스냅샷 (프롬프트에서 언급된 전공군 기반)
    majors = ['친환경·에너지 관련 전공', 'IT 관련 전공', '전통 제조 전공', '기타 전공']
    employ_rates = [88,82,65,70]  # % 단위로 변환 (프롬프트 수치화)
    user_majors_df = pd.DataFrame({'group':majors, 'value':employ_rates})

    # 자동 사이드바 옵션 (기간 필터, 스무딩)
    st.sidebar.header("사용자 데이터 옵션")
    min_year = int(user_jobs_df['date'].dt.year.min())
    max_year = int(user_jobs_df['date'].dt.year.max())
    sel_start = st.sidebar.slider("기간 시작 연도 (사용자 데이터)", min_year, max_year, min_year)
    sel_end = st.sidebar.slider("기간 종료 연도 (사용자 데이터)", min_year, max_year, max_year)
    smoothing = st.sidebar.checkbox("시계열 스무딩(이동평균)", value=False)
    window = st.sidebar.slider("스무딩 윈도우(연)", 2, 5, 2) if smoothing else None

    # 필터 적용
    uj = user_jobs_df.copy()
    uj['year'] = uj['date'].dt.year
    uj = uj[(uj['year'] >= sel_start) & (uj['year'] <= sel_end)].sort_values(['group','date'])

    if smoothing and window:
        uj['value_smooth'] = uj.groupby('group')['value'].transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        y_col = 'value_smooth'
    else:
        y_col = 'value'

    st.subheader("녹색 전환에 따른 일자리 지표 (프롬프트 기반 예시)")
    fig_u1 = px.line(uj, x='date', y=y_col, color='group', markers=True,
                     labels={'date':'연도','value':'지표','value_smooth':'스무딩 지표','group':'구분'},
                     title='녹색 일자리 증가 vs 화석연료 기반 일자리 감소 (예시)')
    st.plotly_chart(fig_u1, use_container_width=True)
    st.markdown("설명: 보고서 본문에 기술된 내용을 바탕으로 예시 수치를 생성한 시계열입니다.")

    st.subheader("전공별 취업률(스냅샷, 예시)")
    fig_u2 = px.bar(user_majors_df, x='group', y='value', text='value',
                    labels={'group':'전공','value':'취업률(%)'}, title='전공별 취업률(예시)')
    st.plotly_chart(fig_u2, use_container_width=True)
    st.markdown("설명: 보고서에서 '친환경·에너지 관련 전공자의 취업률이 평균보다 높다'는 문장을 수치화한 예시입니다.")

    st.download_button("사용자 입력 데이터(일자리) CSV 다운로드", user_jobs_df.to_csv(index=False).encode('utf-8-sig'), "user_jobs.csv", "text/csv")
    st.download_button("사용자 입력 데이터(전공별 취업률) CSV 다운로드", user_majors_df.to_csv(index=False).encode('utf-8-sig'), "user_majors.csv", "text/csv")

    st.markdown("---")
    st.subheader("리포트 기반 권고 (요약)")
    st.write("""
    1. 녹색 전환 관련 역량(녹색기술, 에너지, 데이터 분석)을 키우세요.  
    2. 전통 산업 축소에 대비해 재교육·전환을 준비하세요.  
    3. 학교/동아리에서 기후-취업 연계 프로젝트를 시도해 실무 경험을 쌓으세요.
    """)

# ---------------------------
# 개발자용 안내: Kaggle API 인증 등
# ---------------------------
with st.expander("개발자(또는 실행환경)용: Kaggle API 및 추가 안내"):
    st.markdown("""
    - Kaggle 데이터 사용 시:
      1. `pip install kaggle`
      2. Kaggle 계정 > Account > Create API token -> `kaggle.json` 다운로드
      3. Codespaces/로컬: `~/.kaggle/kaggle.json` 위치에 저장 (`chmod 600 ~/.kaggle/kaggle.json`)
      4. 예: `kaggle datasets download -d <owner/dataset>` 후 압축 해제
    - 주의: 이 앱은 기본적으로 NASA/WorldBank 공개 API를 우선 시도합니다. API 실패 시 예시 데이터로 자동 전환됩니다.
    """)

# ---------------------------
# 끝
# ---------------------------
