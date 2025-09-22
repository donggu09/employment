"""
Streamlit Dashboard (Korean) - V24 (Optimized Loading)
This version optimizes the initial data loading process by fetching all external APIs concurrently, significantly reducing the startup time of the application.

- Core Topic: 'The Impact of Climate Change on Employment'
- Key Upgrades:
  1) **Concurrent API Calls**: Implemented `concurrent.futures.ThreadPoolExecutor` to fetch the three main data sources (NASA, NOAA, World Bank) in parallel instead of sequentially.
  2) **Reduced Timeout**: The default timeout for individual API requests in the `retry_get` function has been reduced from 30 to 15 seconds to fail faster on unresponsive servers.
  3) **Improved User Experience**: The initial loading spinner now reflects the parallel fetching process, and the overall time to see the dashboard is much shorter.
"""

import io
import time
import datetime
import os
import json
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from streamlit_lottie import st_lottie
import openpyxl # For Excel file handling

# = a============================================================================
# 0. CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="기후 위기는 환경을 넘어 취업까지 흔든다",
    page_icon="🌍",
    layout="wide"
)

# --- Custom CSS for Dark Mode UI ---
st.markdown("""
<style>
    /* Main background color set to dark */
    .stApp {
        background-color: #1E1E1E;
        color: #EAEAEA;
    }
    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    /* Tab styles for dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E; /* Match main background to remove box effect */
        border-radius: 4px 4px 0px 0px;
        border-bottom: 2px solid transparent;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #A0A0A0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
        border-bottom: 2px solid #0078F2; /* Bright blue highlight for active tab */
        color: #FFFFFF;
    }
    /* Metric styling for dark mode */
    div[data-testid="stMetricLabel"] {
        display: flex;
        align-items: center;
        color: #A0A0A0; /* Lighter gray for labels */
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF; /* White for values */
    }
    /* Ensure Streamlit widgets have light text */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-1y4p8pa {
        color: #EAEAEA;
    }
</style>
""", unsafe_allow_html=True)


# --- App constants ---
TODAY = datetime.datetime.now().date()
CONFIG = {
    "nasa_gistemp_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "worldbank_api_url": "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS",
    "noaa_co2_url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
    "lottie_home_url": "https://lottie.host/175b5a27-63f5-4220-8374-e32a13f789e9/5N7sBfSbB6.json",
    "lottie_career_game_url": "https://lottie.host/7e05e830-7456-4c31-b844-93b5a1b55909/Rk4yQO6fS3.json"
}
MEMO_FILE = "memos.json"

# --- Global Session with Retry Strategy ---
_SESSION = requests.Session()
_retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=1
)
_adapter = HTTPAdapter(max_retries=_retry_strategy, pool_maxsize=10)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


# ==============================================================================
# 1. UTILITY & DATA FUNCTIONS
# ==============================================================================
def retry_get(url: str, params: Optional[Dict] = None, **kwargs: Any) -> Optional[requests.Response]:
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; StreamlitApp/1.0)'}
    error_message = ""
    try:
        resp = _SESSION.get(url, params=params, headers=headers, timeout=kwargs.get('timeout', 15), allow_redirects=True, verify=True)
        resp.raise_for_status()
        return resp
    except requests.exceptions.ConnectTimeout:
        error_message = f"**API(`{url.split('//')[1].split('/')[0]}`) 연결 시간 초과:** 15초 내에 서버로부터 응답을 받지 못했습니다. 서버가 일시적으로 느리거나, 네트워크 제약 때문일 수 있습니다."
    except requests.exceptions.HTTPError as e:
        error_message = f"**API(`{url.split('//')[1].split('/')[0]}`) 서버 오류:** 서버에서 `{e.response.status_code}` 오류를 반환했습니다."
    except requests.exceptions.RequestException as e:
        error_message = f"**API(`{url.split('//')[1].split('/')[0]}`) 요청 실패:** 인터넷 연결을 확인해주세요. ({e.__class__.__name__})"
    
    if error_message:
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        if error_message not in st.session_state.api_errors:
            st.session_state.api_errors.append(error_message)
    return None

@st.cache_data(ttl=3600)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date'])
        d = d[d['date'].dt.date <= TODAY]
    else:
        return pd.DataFrame()

    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    subset_cols = ['date', 'group'] if 'group' in d.columns else ['date']
    d = d.drop_duplicates(subset=subset_cols)
    sort_cols = ['group', 'date'] if 'group' in d.columns else ['date']
    d = d.sort_values(sort_cols).reset_index(drop=True)
    if 'group' in d.columns:
        d['value'] = d.groupby('group')['value'].transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
    else:
        d['value'] = d['value'].interpolate(method='linear', limit_direction='both')
    return d.dropna(subset=['value']).reset_index(drop=True)

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_gistemp_csv() -> Optional[pd.DataFrame]:
    resp = retry_get(CONFIG["nasa_gistemp_url"])
    if resp is None: return None
    try:
        content = resp.content.decode('utf-8', errors='replace')
        lines = content.split('\n')
        data_start_index = next((i for i, line in enumerate(lines) if line.strip().startswith('Year,')), -1)
        if data_start_index == -1: raise ValueError("CSV Header 'Year,' not found.")
        df = pd.read_csv(io.StringIO("\n".join(lines[data_start_index:])))
        df.columns = [c.strip() for c in df.columns]
        df_long = df.melt(id_vars=['Year'], value_vars=[m for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] if m in df.columns], var_name='Month', value_name='Anomaly')
        month_map = {name: num for num, name in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)}
        df_long['date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'].map(month_map).astype(str), errors='coerce')
        df_final = df_long[['date']].copy()
        df_final['value'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')
        df_final['group'] = '지구 평균 온도 이상치(℃)'
        return df_final.dropna(subset=['date', 'value'])
    except Exception as e:
        if 'api_errors' not in st.session_state: st.session_state.api_errors = []
        if f"**NASA GISTEMP Parsing Error:** `{e}`" not in st.session_state.api_errors: st.session_state.api_errors.append(f"**NASA GISTEMP 데이터 파싱 오류:** `{e}`")
        return None

@st.cache_data(ttl=3600)
def fetch_noaa_co2_data() -> Optional[pd.DataFrame]:
    resp = retry_get(CONFIG["noaa_co2_url"])
    if resp is None: return None
    try:
        df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')), comment='#', delim_whitespace=True, header=None, names=['year', 'month', 'decimal_date', 'average', 'interpolated', 'trend', 'days', 'uncertainty'])
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        df_final = df[['date', 'interpolated']].rename(columns={'interpolated': 'value'})
        df_final['group'] = '대기 중 CO₂ 농도 (ppm)'
        return df_final[df_final['value'] > 0]
    except Exception as e:
        if 'api_errors' not in st.session_state: st.session_state.api_errors = []
        if f"**NOAA CO₂ Parsing Error:** `{e}`" not in st.session_state.api_errors: st.session_state.api_errors.append(f"**NOAA CO₂ 데이터 파싱 오류:** `{e}`")
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment() -> Optional[pd.DataFrame]:
    resp = retry_get(CONFIG["worldbank_api_url"], params={'format': 'json', 'per_page': '20000'})
    if resp is None: return None
    try:
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]: return None
        df = pd.json_normalize(data[1])
        df = df[['country.value', 'countryiso3code', 'date', 'value']]
        df.columns = ['group', 'iso_code', 'year', 'value']
        df['date'] = pd.to_datetime(df['year'] + '-01-01', errors='coerce')
        return df[['date', 'group', 'iso_code', 'value']].dropna()
    except Exception as e:
        if 'api_errors' not in st.session_state: st.session_state.api_errors = []
        if f"**World Bank Parsing Error:** `{e}`" not in st.session_state.api_errors: st.session_state.api_errors.append(f"**World Bank 데이터 파싱 오류:** `{e}`")
        return None

# --- Embedded Sample Data ---
@st.cache_data
def get_sample_climate_data() -> pd.DataFrame:
    csv_data = """date,value,group
2018-01-01,0.85,"지구 평균 온도 이상치(℃) (예시)"
2019-01-01,0.98,"지구 평균 온도 이상치(℃) (예시)"
2020-01-01,1.16,"지구 평균 온도 이상치(℃) (예시)"
2021-01-01,0.86,"지구 평균 온도 이상치(℃) (예시)"
2022-01-01,0.91,"지구 평균 온도 이상치(℃) (예시)"
2023-01-01,1.08,"지구 평균 온도 이상치(℃) (예시)"
2024-01-01,1.35,"지구 평균 온도 이상치(℃) (예시)"
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_data
def get_sample_co2_data() -> pd.DataFrame:
    csv_data = """date,value,group
2018-01-01,408.21,"대기 중 CO₂ 농도 (ppm) (예시)"
2019-01-01,410.92,"대기 중 CO₂ 농도 (ppm) (예시)"
2020-01-01,413.4,"대기 중 CO₂ 농도 (ppm) (예시)"
2021-01-01,415.4,"대기 중 CO₂ 농도 (ppm) (예시)"
2022-01-01,418.28,"대기 중 CO₂ 농도 (ppm) (예시)"
2023-01-01,420.51,"대기 중 CO₂ 농도 (ppm) (예시)"
2024-01-01,423.01,"대기 중 CO₂ 농도 (ppm) (예시)"
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_data
def get_sample_employment_data() -> pd.DataFrame:
    csv_data = """date,group,iso_code,value
2018-01-01,World (예시),WLD,20.21
2020-01-01,World (예시),WLD,20.53
2022-01-01,World (예시),WLD,21.0
2024-01-01,World (예시),WLD,21.4
2018-01-01,Korea (예시),KOR,22.8
2020-01-01,Korea (예시),KOR,23.2
2022-01-01,Korea (예시),KOR,23.7
2024-01-01,Korea (예시),KOR,24.1
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_data
def get_sample_korea_employment_data() -> pd.DataFrame:
    csv_data = """연도,취업자 수 (만 명),실업률 (%)
2019,2712.3,3.8
2020,2690.4,4.0
2021,2727.3,3.7
2022,2808.9,2.9
2023,2841.6,2.8
2024,2869.8,2.8
"""
    return pd.read_csv(io.StringIO(csv_data))

# --- Helper Functions ---
@st.cache_data
def load_lottie_data(url: str):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        return None
    return None

def load_memos():
    try:
        if not os.path.exists(MEMO_FILE):
            with open(MEMO_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(MEMO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_memos(memos):
    with open(MEMO_FILE, "w", encoding="utf-8") as f:
        json.dump(memos, f, ensure_ascii=False, indent=4)

# ==============================================================================
# 2. UI RENDERING FUNCTIONS FOR TABS
# ==============================================================================
def display_home_tab(climate_df, co2_df, employment_df):
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown("<h1 style='text-align: left;'>🌍 기후 위기는<br>환경을 넘어, 취업까지 흔든다</h1>", unsafe_allow_html=True)
        st.markdown("#### 1403 권초현, 1405 김동현, 1410 신수아, 1416 조정모")
        st.markdown("""
        기후변화는 더 이상 먼 미래의 이야기가 아닙니다. 
        우리의 **미래 산업 구조**와 **커리어**를 결정짓는 핵심 변수가 되었습니다.
        
        이 대시보드는 실시간 데이터를 통해 기후 변화가 직업 세계에 미치는 영향을 분석하고,
        미래를 준비하는 청소년들에게 필요한 인사이트와 전략을 제공합니다.
        """)
    with col2:
        lottie_home = load_lottie_data(CONFIG['lottie_home_url'])
        if lottie_home:
            st_lottie(lottie_home, height=300, key="home_lottie")
    
    st.markdown("---")
    st.subheader("📊 대시보드 핵심 지표")
    
    mcol1, mcol2, mcol3 = st.columns(3)
    if not all(df.empty for df in [climate_df, co2_df, employment_df]):
        try:
            latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
            latest_co2 = co2_df.sort_values('date', ascending=False).iloc[0]
            mcol1.metric(label="🌡️ 최신 온도 이상치", value=f"{latest_climate['value']:.2f} ℃", help=f"기준일: {latest_climate['date']:%Y-%m}")
            mcol2.metric(label="☁️ 최신 CO₂ 농도", value=f"{latest_co2['value']:.2f} ppm", help=f"기준일: {latest_co2['date']:%Y-%m}")
            mcol3.metric(label="💼 고용 데이터 국가 수", value=f"{employment_df['group'].nunique()} 개")
        except (IndexError, ValueError, TypeError):
            st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    else:
        st.info("데이터를 불러오는 중이거나 API 호출에 실패했습니다.")

    st.markdown("---")
    st.subheader("🚀 주요 기능 바로가기")
    
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        st.markdown("##### 📊 글로벌 동향 분석")
        st.write("실시간 데이터로 기후와 고용의 큰 그림을 확인해보세요.")
    with qcol2:
        st.markdown("##### 🚀 나의 미래 설계")
        st.write("간단한 시뮬레이션으로 나의 선택이 미래에 미칠 영향을 예측해보세요.")
    with qcol3:
        st.markdown("##### ✍️ 다짐 공유하기")
        st.write("기후 위기 대응을 위한 당신의 작은 실천을 모두와 공유해보세요.")

    st.markdown("---")
    display_data_status()
    display_api_errors()

def display_data_status():
    st.subheader("데이터 출처 현황")
    status = st.session_state.get('data_status', {})
    cols = st.columns(3)
    status_map = {'Live': '🟢 실시간', 'Sample': '🟡 예시'}
    cols[0].markdown(f"**NASA GISTEMP (기온)**: {status_map.get(status.get('climate'), 'N/A')}")
    cols[1].markdown(f"**NOAA CO₂ (이산화탄소)**: {status_map.get(status.get('co2'), 'N/A')}")
    cols[2].markdown(f"**World Bank (고용)**: {status_map.get(status.get('employment'), 'N/A')}")

def display_api_errors():
    if st.session_state.get('api_errors'):
        st.warning("⚠️ 하나 이상의 실시간 데이터 로딩에 실패하여 예시 데이터로 대체되었습니다.", icon="📡")
        with st.expander("상세 오류 정보 보기"):
            for error in st.session_state.api_errors:
                st.error(error, icon="🔥")

def display_global_trends_tab(climate_df, co2_df, employment_df):
    st.subheader("📈 글로벌 동향: 숫자가 말하는 기후와 일자리 변화")
    
    col1, col2, col3 = st.columns(3)
    if not all(df.empty for df in [climate_df, co2_df, employment_df]):
        try:
            latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
            latest_co2 = co2_df.sort_values('date', ascending=False).iloc[0]
            col1.metric(label="🌡️ 최신 온도 이상치", value=f"{latest_climate['value']:.2f} ℃", help=f"기준일: {latest_climate['date']:%Y-%m}")
            col2.metric(label="☁️ 최신 CO₂ 농도", value=f"{latest_co2['value']:.2f} ppm", help=f"기준일: {latest_co2['date']:%Y-%m}")
            col3.metric(label="💼 고용 데이터 국가 수", value=f"{employment_df['group'].nunique()} 개")
        except (IndexError, ValueError, TypeError):
            st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    else:
        st.info("데이터를 불러오는 중이거나 API 호출에 실패했습니다.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🌡️ 지구 평균 온도 이상치")
        if not climate_df.empty:
            fig = px.line(climate_df, x='date', y='value', labels={'date': '', 'value': '온도 이상치 (°C)'}, color_discrete_sequence=['#d62728'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### 💨 대기 중 CO₂ 농도 (마우나로아)")
        if not co2_df.empty:
            fig = px.line(co2_df, x='date', y='value', labels={'date': '', 'value': 'CO₂ (ppm)'}, color_discrete_sequence=['#1f77b4'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA')
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    st.markdown("##### 🏭 산업별 고용 비율 변화")
    if not employment_df.empty:
        employment_df['year'] = pd.to_datetime(employment_df['date']).dt.year
        min_year, max_year = int(employment_df['year'].min()), int(employment_df['year'].max())
        selected_year = st.slider("연도 선택:", min_year, max_year, max_year, key="map_year_slider")
        
        map_df = employment_df[employment_df['year'] == selected_year]
        if not map_df.empty:
            fig_map = px.choropleth(map_df, locations="iso_code", color="value", hover_name="group", color_continuous_scale=px.colors.sequential.Plasma, labels={'value': '고용 비율 (%)'}, title=f"{selected_year}년 전 세계 산업 고용 비율")
            fig_map.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', geo=dict(bgcolor='rgba(0,0,0,0)'), font_color='#EAEAEA')
            st.plotly_chart(fig_map, use_container_width=True)

        all_countries = sorted(employment_df['group'].unique())
        default_countries = [c for c in ['World', 'Korea, Rep.', 'World (예시)', 'Korea (예시)'] if c in all_countries] or all_countries[:2]
        selected_countries = st.multiselect("국가별 추이 비교:", all_countries, default=default_countries)
        if selected_countries:
            comp_df = employment_df[employment_df['group'].isin(selected_countries)]
            fig_comp = px.line(comp_df, x='year', y='value', color='group', labels={'year':'연도', 'value':'산업 고용 비율(%)', 'group':'국가'})
            fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA', legend=dict(font=dict(color='#EAEAEA')))
            st.plotly_chart(fig_comp, use_container_width=True)

def display_analysis_tab(climate_df, co2_df, employment_df):
    st.subheader("🔍 심층 분석: 데이터로 관계 들여다보기")
    
    st.markdown("##### 🔄 기후 지표 vs. 글로벌 산업 고용 상관관계")
    if any(df.empty for df in [climate_df, co2_df, employment_df]):
        st.warning("상관관계를 분석하기 위한 데이터가 부족합니다.")
    else:
        try:
            climate_df['year'] = pd.to_datetime(climate_df['date']).dt.year
            c_ann_agg = climate_df.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})
            co2_df['year'] = pd.to_datetime(co2_df['date']).dt.year
            co2_ann_agg = co2_df.groupby('year')['value'].mean().reset_index().rename(columns={'value':'co2_ppm'})
            employment_df['year'] = pd.to_datetime(employment_df['date']).dt.year
            e_ann_agg = employment_df.groupby(['year'])['value'].median().reset_index().rename(columns={'value':'employment_median'})
            
            merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')
            merged = pd.merge(merged, co2_ann_agg, on='year', how='inner')

            if len(merged) < 2:
                st.warning("데이터 기간이 짧아 상관관계를 분석할 수 없습니다.")
            else:
                corr_col1, corr_col2 = st.columns(2)
                corr_choice = corr_col1.selectbox("비교할 기후 지표:", ('온도 이상치', 'CO₂ 농도'))
                normalize = corr_col2.checkbox("데이터 정규화 (추세 비교)", help="단위가 다른 두 데이터를 0~1 사이 값으로 변환하여 추세 비교를 용이하게 합니다.")
                
                x_var = 'temp_anomaly' if corr_choice == '온도 이상치' else 'co2_ppm'
                y_var = 'employment_median'
                
                plot_df = merged[['year', x_var, y_var]].copy()
                correlation = plot_df[x_var].corr(plot_df[y_var])
                st.metric(f"{corr_choice} vs. 고용 비율 상관계수", f"{correlation:.3f}")

                if normalize:
                    plot_df[x_var] = (plot_df[x_var] - plot_df[x_var].min()) / (plot_df[x_var].max() - plot_df[x_var].min())
                    plot_df[y_var] = (plot_df[y_var] - plot_df[y_var].min()) / (plot_df[y_var].max() - plot_df[y_var].min())
                
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[x_var], name=corr_choice, line=dict(color='#d62728')))
                fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[y_var], name='산업 고용(전세계 중앙값)', yaxis='y2', line=dict(color='#1f77b4')))
                fig_corr.update_layout(xaxis_title="연도", yaxis_title=f"{corr_choice}" if not normalize else "정규화된 값", yaxis2=dict(title="산업 고용 비율 (%)" if not normalize else "정규화된 값", overlaying="y", side="right"), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA', legend_font_color='#EAEAEA')
                st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    st.markdown("---")

    st.markdown("##### 🇰🇷 국내 데이터 심층 분석 (e-나라지표 샘플)")
    st.info("e-나라지표의 '취업자 및 실업자' 통계 샘플 데이터를 활용하여 국내 취업 데이터와 기후 변화의 관계를 분석합니다.")
    
    korea_df = get_sample_korea_employment_data()
    
    temp_yearly = climate_df.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})
    merged_korea = pd.merge(korea_df, temp_yearly, left_on='연도', right_on='year', how='inner')
    
    if len(merged_korea) > 1:
        fig_korea = go.Figure()
        fig_korea.add_trace(go.Scatter(x=merged_korea['연도'], y=merged_korea['실업률 (%)'], name='한국 실업률 (%)', line=dict(color='#ff7f0e')))
        fig_korea.add_trace(go.Scatter(x=merged_korea['연도'], y=merged_korea['temp_anomaly'], name='지구 온도 이상치 (℃)', yaxis='y2', line=dict(color='#d62728')))
        fig_korea.update_layout(title="한국 실업률과 지구 온도 이상치 비교", xaxis_title="연도", yaxis_title="실업률 (%)", yaxis2=dict(title="온도 이상치 (℃)", overlaying="y", side="right"),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA', legend_font_color='#EAEAEA')
        st.plotly_chart(fig_korea, use_container_width=True)
    else:
        st.warning("샘플 데이터와 기후 데이터의 공통 연도가 부족하여 비교 차트를 생성할 수 없습니다.")

def display_job_impact_tab():
    st.subheader("⚖️ 직무 영향 분석: 기회와 위험")
    st.markdown("""
    핵심 원인은 **'녹색 전환(Green Transition)'**입니다. 기후 대응을 위해 사회 전반이 친환경 기술을 도입하면서 새로운 직무가 생겨나고 있기 때문입니다. 
    아래 '직무 전환 탐색기'를 통해 기존 직무가 어떤 기회를 맞이할 수 있는지 알아보세요.
    """)

    job_data = {
        '화력 발전소 기술자': {
            'risk': '매우 높음', 'icon': '🔴',
            'skills': ['발전 설비 운영', '고압 전기 관리', '기계 유지보수'],
            'transitions': {
                '신재생에너지 발전 전문가': ['태양광/풍력 시스템 이해', '에너지 저장 시스템(ESS)'],
                '스마트 그리드 전문가': ['전력망 최적화', '데이터 분석']
            }
        },
        '자동차 내연기관 엔지니어': {
            'risk': '높음', 'icon': '🟠',
            'skills': ['엔진 설계', '열역학', '기계 공학'],
            'transitions': {
                '전기차 배터리 시스템 엔지니어': ['배터리 관리 시스템(BMS)', '전력 전자'],
                '수소연료전지 개발자': ['연료전지 스택 설계', '고압 수소 제어']
            }
        },
        '석탄 광부': {
            'risk': '매우 높음', 'icon': '🔴',
            'skills': ['채굴 기술', '중장비 운용', '안전 관리'],
            'transitions': {
                '지열 에너지 기술자': ['시추 기술', '플랜트 운영'],
                '태양광/풍력 단지 건설 및 유지보수': ['부지 관리', '건설 기술']
            }
        },
        '전통 농업 종사자 (대규모 단일 작물)': {
            'risk': '보통', 'icon': '🟡',
            'skills': ['경작 기술', '병충해 관리', '농기계 운용'],
            'transitions': {
                '스마트팜 운영자': ['데이터 분석', '자동화 시스템 제어', 'IoT 센서 활용'],
                '정밀 농업 컨설턴트': ['GIS/드론 활용', '토양 데이터 분석']
            }
        },
        '석유화학 공장 운영원': {
            'risk': '높음', 'icon': '🟠',
            'skills': ['화학 공정 관리', '안전 관리', '생산 최적화'],
            'transitions': {
                '바이오플라스틱 연구원': ['생분해성 고분자', '바이오매스 처리'],
                '탄소 포집/활용(CCUS) 전문가': ['화학 흡수법', '분리막 기술']
            }
        }
    }

    selected_job = st.selectbox("전환 가능성을 탐색할 직무를 선택하세요:", list(job_data.keys()))

    if selected_job:
        data = job_data[selected_job]
        st.markdown(f"### {selected_job}")
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**전환 위험도**\n\n## {data['icon']} {data['risk']}")
        
        with c2:
            st.markdown("**보유 핵심 역량**")
            for skill in data['skills']:
                st.markdown(f"- {skill}")
        with c3:
            st.markdown("**미래 전환 추천 직무**")
            for job, skills in data['transitions'].items():
                st.markdown(f"**- {job}**")
                st.markdown(f"<small> (필요 역량: {', '.join(skills)})</small>", unsafe_allow_html=True)


def display_career_game_tab():
    st.subheader("🚀 나의 미래 설계하기 (커리어 시뮬레이션)")
    st.info("당신의 선택이 10년 후 커리어와 환경에 어떤 영향을 미치는지 시뮬레이션 해보세요!")
    
    game_col, form_col = st.columns([0.4, 0.6])

    with game_col:
        lottie_career = load_lottie_data(CONFIG['lottie_career_game_url'])
        if lottie_career:
            st_lottie(lottie_career, height=400, key="career_lottie")
        
        st.markdown("""
        ##### 💡 기후 위기를 기회로 바꾸는 전략
        - **데이터 탐구:** 기후와 산업 통계를 분석하며 변화를 예측합니다.
        - **융합 프로젝트:** 자신의 전공과 기후 위기 문제를 연결하는 프로젝트를 수행합니다.
        - **목소리 내기:** 기후 대응과 청년 고용 창출을 연결하여 정책을 제안합니다.
        """)

    with form_col:
        with st.form("career_game_form"):
            st.markdown("##### 🎓 1단계: 대학생")
            major = st.radio("주요 전공:", ("컴퓨터공학 (AI 트랙)", "기계공학", "경제학"), key="major", horizontal=True)
            project = st.radio("졸업 프로젝트:", ("탄소 배출량 예측 AI 모델", "고효율 내연기관 설계", "ESG 경영사례 분석"), key="project")

            st.markdown("##### 💼 2단계: 사회초년생")
            first_job = st.radio("첫 직장:", ("에너지 IT 스타트업", "대기업 정유회사", "금융권 애널리스트"), key="first_job")
            skill_dev = st.radio("핵심 역량 개발:", ("클라우드 기반 데이터 분석", "전통 공정 관리", "재무 분석 및 투자"), key="skill_dev")
            
            submitted = st.form_submit_button("🚀 나의 미래 확인하기")

        if submitted:
            career_score, green_score = 0, 0
            skills = {"데이터 분석":0, "정책/경영":0, "엔지니어링":0, "금융/경제":0}

            if major == "컴퓨터공학 (AI 트랙)": career_score += 20; green_score += 10; skills["데이터 분석"] += 2
            elif major == "기계공학": career_score += 10; skills["엔지니어링"] += 2
            else: career_score += 15; green_score += 5; skills["금융/경제"] += 2
            if project == "탄소 배출량 예측 AI 모델": career_score += 15; green_score += 20; skills["데이터 분석"] += 1; skills["정책/경영"] += 1
            elif project == "고효율 내연기관 설계": career_score += 5; green_score -= 10; skills["엔지니어링"] += 1
            else: career_score += 10; green_score += 10; skills["정책/경영"] += 1; skills["금융/경제"] += 1
            if first_job == "에너지 IT 스타트업": career_score += 15; green_score += 20
            elif first_job == "대기업 정유회사": career_score += 20; green_score -= 10
            else: career_score += 15; green_score += 5
            if skill_dev == "클라우드 기반 데이터 분석": career_score += 20; green_score += 10; skills["데이터 분석"] += 2
            elif skill_dev == "전통 공정 관리": career_score += 10; green_score -= 5; skills["엔지니어링"] += 1
            else: career_score += 15; skills["금융/경제"] += 1
            
            if green_score >= 50 and career_score >= 70: job_title = "기후 기술 최고 전문가"
            elif green_score >= 30 and career_score >= 60: job_title = "그린 에너지 전략가"
            else: job_title = "미래 준비형 인재"

            st.markdown("##### 🎉 최종 결과: 당신의 커리어 카드")
            res1, res2 = st.columns([0.6, 0.4])
            with res1:
                st.markdown(f"#### 💼 **직업:** {job_title}")
                st.metric("🚀 미래 전망 점수", f"{career_score} / 100")
                st.metric("🌱 환경 기여도 점수", f"{green_score} / 75")
            with res2:
                df_skills = pd.DataFrame(dict(r=list(skills.values()), theta=list(skills.keys())))
                fig = px.line_polar(df_skills, r='r', theta='theta', line_close=True, range_r=[0,5], title="나의 역량 레이더 차트")
                fig.update_layout(
                    polar=dict(
                        bgcolor = 'rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(color='#EAEAEA')),
                        angularaxis=dict(tickfont=dict(size=12, color='#EAEAEA'))
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=60, r=60, t=80, b=60),
                    font_color='#EAEAEA'
                )
                st.plotly_chart(fig, use_container_width=True)

def display_memo_board_tab():
    st.subheader("✍️ 나의 실천 다짐 남기기 (공유 방명록)")
    st.markdown("기후 위기 대응을 위한 여러분의 다짐을 남겨주세요! 모든 방문자에게 공유됩니다.")
    
    with st.form("memo_form"):
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            name = st.text_input("닉네임", placeholder="자신을 표현하는 멋진 닉네임을 적어주세요!", key="memo_name")
            memo = st.text_area("실천 다짐", placeholder="예) 텀블러 사용하기, 가까운 거리는 걸어다니기 등", key="memo_text")
        with cols[1]:
            color = st.color_picker("메모지 색상 선택", "#FFFACD", key="memo_color")
            submitted = st.form_submit_button("다짐 남기기!", use_container_width=True)
            if submitted:
                if name and memo:
                    all_memos = load_memos()
                    all_memos.insert(0, {"name": name, "memo": memo, "color": color, "timestamp": str(datetime.datetime.now())})
                    save_memos(all_memos)
                    st.balloons()
                    st.success("소중한 다짐이 모두에게 공유되었습니다!")
                else:
                    st.warning("닉네임과 다짐을 모두 입력해주세요!")
    st.markdown("---")

    st.markdown("##### 💬 우리의 다짐들")
    memos_list = load_memos()
    
    if not memos_list:
        st.info("아직 작성된 다짐이 없어요. 첫 번째 다짐을 남겨주세요!")
    else:
        memo_cols = st.columns(3)
        for i, m in enumerate(memos_list):
            with memo_cols[i % 3]:
                st.markdown(f"""
                <div style="background-color:{m.get('color', '#FFFACD')}; border-left: 5px solid #FF6347; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 180px;">
                    <p style="font-size: 1.1em; color: black; margin-bottom: 10px;">"{m.get('memo', '')}"</p>
                    <strong style="font-size: 0.9em; color: #555;">- {m.get('name', '')} -</strong>
                </div>
                """, unsafe_allow_html=True)

def display_survey_tab():
    st.subheader("📝 설문 및 의견")
    st.markdown("기후 변화와 미래 직업에 대한 여러분의 소중한 의견을 들려주세요!")

    with st.form("survey_form"):
        st.markdown("##### 개인 인식 및 행동")
        q1 = st.radio("1️⃣ 기후변화가 나의 직업(또는 미래 직업)에 영향을 줄 것이라 생각하시나요?", ["매우 그렇다", "조금 그렇다", "별로 아니다", "전혀 아니다"])
        q2 = st.radio("2️⃣ 기후변화 위기의 심각성을 어느 정도로 느끼시나요?", ["매우 심각하다", "어느 정도 심각하다", "보통이다", "심각하지 않다"])
        q3 = st.multiselect("3️⃣ 평소에 실천하고 있는 친환경 활동이 있다면 모두 선택해주세요.", ["분리수거 철저히 하기", "대중교통/자전거 이용", "일회용품 사용 줄이기", "에너지 절약(콘센트 뽑기 등)", "채식/육류 소비 줄이기", "없음"])

        st.markdown("##### 직업 및 교육")
        q4 = st.selectbox("4️⃣ 가장 유망하다고 생각하는 녹색 일자리 분야는 무엇인가요?", ["신재생에너지", "ESG 컨설팅", "친환경 소재 개발", "기후 데이터 분석", "전기/수소차"])
        q5 = st.slider("5️⃣ 미래 직업을 위해 기후변화 관련 역량을 키울 의향이 어느 정도인가요? (0~10점)", 0, 10, 7)
        q6 = st.multiselect("6️⃣ 녹색 일자리 전환을 위해 가장 필요하다고 생각하는 지원은 무엇인가요? (중복 선택 가능)", ["전문 재교육 프로그램", "정부의 재정 지원", "기업의 채용 연계", "진로 멘토링 및 상담"])

        st.markdown("##### 기업 및 사회 정책")
        q7 = st.radio("7️⃣ 기후변화 대응을 위해 기업의 역할이 얼마나 중요하다고 생각하시나요?", ["매우 중요하다", "중요하다", "보통이다", "중요하지 않다"])
        q8 = st.radio("8️⃣ 기후변화 대응을 위한 세금(탄소세 등) 추가 부담에 동의하시나요?", ["적극 찬성", "찬성하는 편", "반대하는 편", "적극 반대"])
        
        submitted = st.form_submit_button("설문 제출하기")

    if submitted:
        st.success("✅ 설문에 참여해주셔서 감사합니다! 아래는 나의 응답 요약입니다.")
        st.markdown("##### 📋 나의 응답 요약")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**직업 영향 인식:** {q1}")
            st.write(f"**위기 심각성 인식:** {q2}")
            st.write(f"**유망 녹색 일자리:** {q4}")
            st.write(f"**기업 역할 중요도:** {q7}")
            st.write(f"**탄소세 동의:** {q8}")
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = q5,
                title = {'text': "나의 역량 개발 의지 점수"},
                gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "#2ca02c"}}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#EAEAEA')
            st.plotly_chart(fig, use_container_width=True)

def display_learn_more_tab():
    st.subheader("📚 더 알아보기: 관련 정보 및 사이트")
    
    st.markdown("##### 💼 녹색 일자리 채용 정보")
    st.markdown("""
    - [워크넷 - 녹색 일자리](https://www.work.go.kr/greenWork/main.do)
    - [환경부 환경산업기술원 - 환경일자리](https://www.job.keiti.re.kr/)
    - [인크루트 - 녹색금융/산업 채용관](https://green.incruit.com/)
    """)
    st.markdown("---")

    st.markdown("##### 🎓 교육 및 학습 자료")
    st.markdown("""
    - [K-MOOC - 기후변화 관련 강좌](http://www.kmooc.kr/search?query=%EA%B8%B0%ED%9B%84%EB%B3%80%ED%99%94)
    - [환경교육포털](https://www.keep.go.kr/portal/1)
    """)
    st.markdown("---")

    st.markdown("##### 📊 데이터 및 보고서 출처")
    st.markdown("""
    - [NASA: GISS Surface Temperature Analysis](https://data.giss.nasa.gov/gistemp/)
    - [NOAA: Global Monitoring Laboratory - CO₂ Data](https://gml.noaa.gov/ccgg/trends/)
    - [The World Bank: Data](https://data.worldbank.org/)
    - [e-나라지표](https://www.index.go.kr/)
    """)

# ==============================================================================
# 3. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_status = {}
        st.session_state.api_errors = []

        with st.spinner("실시간 데이터를 병렬로 빠르게 불러오는 중입니다..."):
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_climate = executor.submit(fetch_gistemp_csv)
                future_co2 = executor.submit(fetch_noaa_co2_data)
                future_employment = executor.submit(fetch_worldbank_employment)

                climate_raw = future_climate.result()
                co2_raw = future_co2.result()
                wb_employment_raw = future_employment.result()
            
            st.session_state.data_status['climate'] = 'Live' if climate_raw is not None and not climate_raw.empty else 'Sample'
            st.session_state.climate_df = preprocess_dataframe(climate_raw if st.session_state.data_status['climate'] == 'Live' else get_sample_climate_data())

            st.session_state.data_status['co2'] = 'Live' if co2_raw is not None and not co2_raw.empty else 'Sample'
            st.session_state.co2_df = preprocess_dataframe(co2_raw if st.session_state.data_status['co2'] == 'Live' else get_sample_co2_data())

            st.session_state.data_status['employment'] = 'Live' if wb_employment_raw is not None and not wb_employment_raw.empty else 'Sample'
            st.session_state.employment_df = preprocess_dataframe(wb_employment_raw if st.session_state.data_status['employment'] == 'Live' else get_sample_employment_data())
            
            st.session_state.data_loaded = True
            time.sleep(0.5)
            st.rerun()
    
    # --- UI Display ---
    tabs = st.tabs(["🏠 홈", "📊 글로벌 동향", "🔍 심층 분석", "⚖️ 직무 영향 분석", "🚀 나의 미래 설계하기", "✍️ 다짐 공유하기", "📝 설문 및 의견", "📚 더 알아보기"])
    
    with tabs[0]:
        display_home_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df)
    with tabs[1]:
        display_global_trends_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df)
    with tabs[2]:
        display_analysis_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df)
    with tabs[3]:
        display_job_impact_tab()
    with tabs[4]:
        display_career_game_tab()
    with tabs[5]:
        display_memo_board_tab()
    with tabs[6]:
        display_survey_tab()
    with tabs[7]:
        display_learn_more_tab()

if __name__ == "__main__":
    main()

