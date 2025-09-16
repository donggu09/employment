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
  * World Bank API (Employment in industry %): https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS?format=json&per_page=20000
  * (앱 실행 환경에서 API 호출이 실패하면 예시 데이터로 자동 대체)
"""

import os
import io
import time
import datetime
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests

# plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    """지정한 횟수만큼 재시도하는 GET 요청 함수"""
    # 기본 헤더 설정 (브라우저처럼 보이게)
    final_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    if headers:
        final_headers.update(headers)

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=final_headers, timeout=timeout)
            resp.raise_for_status()  # 200번대 응답이 아니면 예외 발생
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(backoff * (attempt + 1))
                continue
            st.sidebar.warning(f"API 요청 실패: {url.split('?')[0]} ({e})")
            return None

# ---------------------------
# 공개 데이터 로드 함수 (캐시)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_gistemp_csv() -> Optional[pd.DataFrame]:
    """
    NASA GISTEMP (global monthly anomalies) 로드 및 파싱
    출처 주석: https://data.giss.nasa.gov/gistemp/
    CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    resp = retry_get(url, max_retries=1)
    if resp is None:
        return None
    content = resp.content.decode('utf-8', errors='replace')
    try:
        lines = content.split('\n')
        # 파일 상단 주석을 건너뛰고 실제 헤더('Year', 'Jan'...)를 찾음
        data_start_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('Year,'):
                data_start_index = i
                break

        if data_start_index == -1:
            st.sidebar.error("GISTEMP CSV에서 데이터 헤더를 찾지 못했습니다.")
            return None

        # 유효한 데이터 부분만 읽기
        valid_csv_data = "\n".join(lines[data_start_index:])
        df = pd.read_csv(io.StringIO(valid_csv_data))

        # 컬럼 이름 공백 제거
        df.columns = [c.strip() for c in df.columns]

        # 월(Jan-Dec) 데이터만 사용하여 long format으로 변환
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        present_months = [m for m in months if m in df.columns]
        if not present_months:
            return None

        df_long = df.melt(id_vars=['Year'], value_vars=present_months, var_name='Month', value_name='Anomaly')

        # Year와 Month를 사용하여 date 객체 생성
        month_map = {name: num for num, name in enumerate(months, 1)}
        def make_date(row):
            try:
                y = int(row['Year'])
                m = month_map.get(row['Month'])
                return datetime.date(y, m, 1)
            except (ValueError, TypeError):
                return None

        df_long['date'] = df_long.apply(make_date, axis=1)
        df_long = df_long.dropna(subset=['date'])

        # 'value' 컬럼을 숫자로 변환
        df_long['value'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')

        # 표준 형식으로 컬럼 정리
        df_long = df_long[['date','value']]

        # 미래 날짜 데이터 제거 및 정렬
        df_long = df_long[df_long['date'] <= TODAY]
        df_long = df_long.dropna(subset=['value']).sort_values('date').reset_index(drop=True)

        # 그룹명 지정
        df_long['group'] = '지구 평균 온도 이상치(℃)'
        return df_long
    except Exception as e:
        st.sidebar.error(f"GISTEMP 데이터 파싱 중 오류: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment() -> Optional[pd.DataFrame]:
    """
    World Bank API: Employment in industry (% of total employment)
    API: https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS
    출처 주석: https://data.worldbank.org/indicator/SL.IND.EMPL.ZS
    """
    api = "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS"
    params = {'format':'json','per_page':'20000'} # per_page는 문자열로 전달하는 것이 안전
    resp = retry_get(api, params=params, max_retries=1)
    if resp is None:
        return None
    try:
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            st.sidebar.warning("World Bank API 응답이 비어있습니다.")
            return None

        records = data[1]
        rows = []
        for rec in records:
            country = rec.get('country', {}).get('value')
            year_str = rec.get('date')
            val = rec.get('value')

            if not all([country, year_str, val]):
                continue

            try:
                y = int(year_str)
                d = datetime.date(y, 1, 1)
                if d <= TODAY:
                    rows.append({'date': d, 'group': country, 'value': float(val)})
            except (ValueError, TypeError):
                continue

        if not rows:
            return None

        return pd.DataFrame(rows)
    except Exception as e:
        st.sidebar.error(f"World Bank 데이터 처리 중 오류: {e}")
        return None

# ---------------------------
# 예시(대체) 데이터 (공개 데이터 호출 실패 시 사용)
# ---------------------------
def sample_climate_data() -> pd.DataFrame:
    years = list(range(TODAY.year - 14, TODAY.year + 1))
    dates = [datetime.date(y, 1, 1) for y in years]
    values = np.round(np.linspace(0.4, 1.1, len(years)) + np.random.normal(0, 0.05, len(years)), 3)
    df = pd.DataFrame({'date': dates, 'value': values})
    df['group'] = '지구 평균 온도 이상치(℃)'
    return df

def sample_employment_data() -> pd.DataFrame:
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
def preprocess_dataframe(df: pd.DataFrame, is_climate_data: bool = False) -> pd.DataFrame:
    """데이터프레임 전처리 공통 함수"""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # 1. 날짜 형식 통일 및 미래 날짜 제거
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d = d[d['date'].dt.date <= TODAY]

    # 2. 값 형식 통일
    d['value'] = pd.to_numeric(d['value'], errors='coerce')

    # 3. 중복 제거
    subset_cols = ['date', 'group'] if 'group' in d.columns else ['date']
    d = d.drop_duplicates(subset=subset_cols)

    # 4. 결측치 처리 (보간법)
    if 'group' in d.columns:
        d = d.sort_values(['group','date']).reset_index(drop=True)
        # transform을 사용하여 그룹별 보간 적용
        d['value'] = d.groupby('group')['value'].transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
    else:
        d = d.sort_values('date').reset_index(drop=True)
        d['value'] = d['value'].interpolate(method='linear', limit_direction='both')

    # 5. 최종 결측치 제거 및 인덱스 리셋
    d = d.dropna(subset=['value']).reset_index(drop=True)

    return d

# ---------------------------
# 공개 데이터 불러오기 및 처리
# ---------------------------
st.sidebar.title("데이터 로드 상태")
with st.spinner("공식 공개 데이터를 불러오는 중입니다..."):
    climate_raw = fetch_gistemp_csv()
    emp_raw = fetch_worldbank_employment()

# Fallback to sample data if fetching failed
if climate_raw is None:
    st.sidebar.error("NASA GISTEMP 로드 실패 → 예시 데이터 사용")
    climate_raw = sample_climate_data()
else:
    st.sidebar.success("NASA GISTEMP 로드 성공")

if emp_raw is None:
    st.sidebar.error("World Bank 데이터 로드 실패 → 예시 데이터 사용")
    emp_raw = sample_employment_data()
else:
    st.sidebar.success("World Bank 데이터 로드 성공")

# Preprocess data
climate_df = preprocess_dataframe(climate_raw, is_climate_data=True)
employment_df = preprocess_dataframe(emp_raw)

# ---------------------------
# UI: 탭 구조
# ---------------------------
tab1, tab2 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력(리포트 기반) 대시보드"])

with tab1:
    st.header("공식 공개 데이터 기반 분석")
    st.markdown("**출처**: NASA GISTEMP (기후 이상치), World Bank (산업별 고용 비율). API 호출 실패 시 예시 데이터로 자동 대체됩니다.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("지구 평균 온도 이상치 변화")
        if not climate_df.empty:
            fig = px.line(climate_df, x='date', y='value', title='지구 평균 온도 이상치 (월별)', labels={'date':'날짜','value':'이상치 (℃)'})
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("기후(전처리) CSV 다운로드", climate_df.to_csv(index=False, encoding='utf-8-sig'), "climate_preprocessed.csv", "text/csv")
        else:
            st.warning("표시할 기후 데이터가 없습니다.")

    with col_right:
        st.subheader("최신 산업별 고용 비율 (상위 10개)")
        if not employment_df.empty:
            try:
                employment_df['year'] = employment_df['date'].dt.year
                latest_year = int(employment_df['year'].max())
                latest_df = employment_df[employment_df['year'] == latest_year]
                display_table = latest_df.sort_values('value', ascending=False).head(10)
                display_table = display_table[['group', 'value']].rename(columns={'group':'국가/그룹','value':f'산업 고용 비율(%, {latest_year}년)'})
                st.dataframe(display_table.reset_index(drop=True))
            except Exception:
                st.write("고용 데이터 요약을 생성하지 못했습니다.")
            st.download_button("고용(전처리) CSV 다운로드", employment_df.to_csv(index=False, encoding='utf-8-sig'), "employment_preprocessed.csv", "text/csv")
        else:
            st.warning("표시할 고용 데이터가 없습니다.")

    st.markdown("---")
    st.subheader("기후(온도 이상치) vs 산업 고용(연 단위 비교)")
    try:
        if not climate_df.empty and not employment_df.empty:
            c_ann = climate_df.copy()
            c_ann['year'] = c_ann['date'].dt.year
            c_ann_agg = c_ann.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})

            e_ann = employment_df.copy()
            e_ann['year'] = e_ann['date'].dt.year
            e_ann_agg = e_ann.groupby('year')['value'].median().reset_index().rename(columns={'value':'industry_employment_median'})

            merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')
            if merged.empty:
                st.info("공개 데이터 간 공통 연도가 없어 비교 그래프를 생성할 수 없습니다. (두 데이터셋 모두 예시 데이터일 수 있습니다.)")
            else:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=merged['year'], y=merged['temp_anomaly'], mode='lines', name='기후 이상치(연평균)', yaxis='y1'))
                fig_comp.add_trace(go.Scatter(x=merged['year'], y=merged['industry_employment_median'], mode='lines', name='산업 고용(연도별 중앙값)', yaxis='y2'))
                fig_comp.update_layout(
                    title="연도별 기후 이상치와 산업 고용 비율 비교",
                    xaxis_title="연도",
                    yaxis=dict(title="기후 이상치(℃)"),
                    yaxis2=dict(title="산업 고용 비율(%)", overlaying='y', side='right', showgrid=False),
                    legend=dict(x=0.1, y=0.9)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("비교 분석을 위한 데이터가 부족합니다.")
    except Exception as e:
        st.error(f"비교 그래프 생성 중 오류 발생: {e}")

with tab2:
    st.header("사용자 입력(프롬프트 기반) 대시보드")
    st.markdown("입력: 사용자가 제공한 보고서 텍스트(프롬프트)만을 바탕으로 생성한 예시 데이터입니다. 앱 실행 중 파일 업로드나 추가 텍스트 입력을 요구하지 않습니다.")

    # --- 사용자 데이터 생성 (프롬프트 기반) ---
    years = list(range(2018, min(TODAY.year, 2024) + 1))
    if not years: years = [TODAY.year]

    # 1) 녹색 일자리 vs 화석연료 일자리
    green_jobs = [5, 8, 12, 15, 18, 22, 26][:len(years)]
    fossil_jobs = [0, -2, -4, -6, -9, -12, -15][:len(years)]
    dates = [datetime.date(y, 1, 1) for y in years]

    user_jobs_df = pd.DataFrame({
        'date': dates * 2,
        'group': ['녹색 일자리 증가'] * len(years) + ['화석연료 기반 일자리 감소'] * len(years),
        'value': green_jobs + fossil_jobs
    })
    user_jobs_df['date'] = pd.to_datetime(user_jobs_df['date'])

    # 2) 전공별 취업률 스냅샷
    majors = ['친환경·에너지', 'IT·데이터', '전통 제조업', '기타']
    employ_rates = [88, 82, 65, 70]
    user_majors_df = pd.DataFrame({'group': majors, 'value': employ_rates})

    # --- 사이드바 옵션 ---
    st.sidebar.header("사용자 데이터 옵션")
    min_year, max_year = user_jobs_df['date'].dt.year.min(), user_jobs_df['date'].dt.year.max()
    sel_start, sel_end = st.sidebar.slider("기간 선택 (사용자 데이터)", min_year, max_year, (min_year, max_year))

    smoothing = st.sidebar.checkbox("시계열 스무딩(이동평균)", value=False)
    window = st.sidebar.slider("스무딩 윈도우(연)", 2, 5, 2) if smoothing else 2

    # --- 필터링 및 데이터 처리 ---
    uj_filtered = user_jobs_df[
        (user_jobs_df['date'].dt.year >= sel_start) &
        (user_jobs_df['date'].dt.year <= sel_end)
    ].sort_values(['group','date'])

    y_col = 'value'
    if smoothing:
        uj_filtered['value_smooth'] = uj_filtered.groupby('group')['value'].transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        y_col = 'value_smooth'


    st.subheader("녹색 전환에 따른 일자리 지표 (프롬프트 기반 예시)")
    fig_u1 = px.line(uj_filtered, x='date', y=y_col, color='group', markers=True,
                     labels={'date':'연도', y_col:'변화 지표', 'group':'구분'},
                     title='녹색 일자리 증가 vs 화석연료 기반 일자리 감소')
    st.plotly_chart(fig_u1, use_container_width=True)
    st.markdown("설명: 보고서 본문에 기술된 내용을 바탕으로 예시 수치를 생성한 시계열입니다.")

    st.subheader("전공별 취업률 (스냅샷 예시)")
    fig_u2 = px.bar(user_majors_df, x='group', y='value', text='value',
                    labels={'group':'전공 분야','value':'취업률(%)'}, title='전공 분야별 예상 취업률')
    fig_u2.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_u2, use_container_width=True)
    st.markdown("설명: '친환경·에너지 전공자의 취업률이 높다'는 문장을 수치화한 예시입니다.")

    st.download_button("일자리 지표(예시) CSV 다운로드", user_jobs_df.to_csv(index=False, encoding='utf-8-sig'), "user_jobs_sample.csv", "text/csv")
    st.download_button("전공별 취업률(예시) CSV 다운로드", user_majors_df.to_csv(index=False, encoding='utf-8-sig'), "user_majors_sample.csv", "text/csv")

    st.markdown("---")
    st.subheader("리포트 기반 권고 사항 (요약)")
    st.write("""
    - **역량 강화**: 녹색 기술, 신재생에너지, 데이터 분석 등 미래 유망 분야의 전문성을 키우세요.
    - **전환 준비**: 전통 산업의 변화에 대비하여 재교육 및 직무 전환 프로그램을 적극적으로 탐색하세요.
    - **실무 경험**: 기후 변화와 직업을 연계한 프로젝트나 인턴십에 참여하여 실질적인 경험을 쌓으세요.
    """)

# ---------------------------
# 개발자용 안내
# ---------------------------
with st.expander("개발자 및 실행 환경 참고사항"):
    st.markdown("""
    - **Kaggle 데이터 사용 시**:
      1. `pip install kaggle`
      2. Kaggle 계정 > Account > Create API token -> `kaggle.json` 다운로드
      3. 로컬 환경의 `~/.kaggle/kaggle.json` 위치에 저장 (`chmod 600 ~/.kaggle/kaggle.json`)
      4. 예: `kaggle datasets download -d <owner/dataset>` 실행 후 압축 해제
    - **주의**: 이 앱은 NASA/WorldBank 공개 API를 우선적으로 호출하며, 실패 시에만 내장된 예시 데이터로 자동 전환됩니다.
    """)
