"""
Streamlit Dashboard (Korean) - ENHANCED
- Topic: 'The Impact of Climate Change on Employment'
- Features:
  1) Interactive dashboards with public data (NASA GISTEMP, World Bank).
  2) Simulated dashboard based on a text prompt.
- Enhancements:
  - Added Choropleth map for global employment data.
  - Added multi-country selector for comparison.
  - Added data normalization option for dual-axis chart.
  - Added moving average trendline for climate data.
- Implementation Rules:
  - Data Standardization: date, value, group (optional).
  - Preprocessing: Handle missing values, type conversion, duplicates, future data.
  - Caching: Use @st.cache_data.
  - CSV Download Button.
  - All UI in Korean.
"""

import os
import io
import time
import datetime
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# 0. CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="기후와 취업: 데이터 대시보드",
    page_icon="🌍",
    layout="wide"
)

# --- App constants ---
TODAY = datetime.datetime.now().date()
CONFIG = {
    "nasa_gistemp_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "worldbank_api_url": "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS",
    "font_path": "/fonts/Pretendard-Bold.ttf",
}

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================
def retry_get(url: str, params: Optional[Dict] = None, **kwargs: Any) -> Optional[requests.Response]:
    """Robust GET request with retries and user-agent."""
    final_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for attempt in range(kwargs.get('max_retries', 2) + 1):
        try:
            resp = requests.get(url, params=params, headers=final_headers, timeout=kwargs.get('timeout', 15))
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < kwargs.get('max_retries', 2):
                time.sleep(kwargs.get('backoff', 1.0) * (attempt + 1))
                continue
            st.sidebar.warning(f"API 요청 실패: {url.split('?')[0]} ({e})")
            return None

@st.cache_data(ttl=3600)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and preprocess a dataframe."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date'])
    d = d[d['date'].dt.date <= TODAY]
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    subset_cols = ['date', 'group'] if 'group' in d.columns else ['date']
    d = d.drop_duplicates(subset=subset_cols)
    sort_cols = ['group', 'date'] if 'group' in d.columns else ['date']
    d = d.sort_values(sort_cols).reset_index(drop=True)
    if 'group' in d.columns:
        d['value'] = d.groupby('group')['value'].transform(lambda s: s.interpolate(method='linear', limit_direction='both', limit_area='inside'))
    else:
        d['value'] = d['value'].interpolate(method='linear', limit_direction='both', limit_area='inside')
    return d.dropna(subset=['value']).reset_index(drop=True)

# ==============================================================================
# 2. DATA LOADING & PROCESSING (with Caching)
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_gistemp_csv() -> Optional[pd.DataFrame]:
    """Fetch and parse NASA GISTEMP global monthly anomalies."""
    resp = retry_get(CONFIG["nasa_gistemp_url"], max_retries=1)
    if resp is None: return None
    try:
        content = resp.content.decode('utf-8', errors='replace')
        lines = content.split('\n')
        data_start_index = next((i for i, line in enumerate(lines) if line.strip().startswith('Year,')), -1)
        if data_start_index == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(lines[data_start_index:])))
        df.columns = [c.strip() for c in df.columns]
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        present_months = [m for m in months if m in df.columns]
        df_long = df.melt(id_vars=['Year'], value_vars=present_months, var_name='Month', value_name='Anomaly')
        month_map = {name: num for num, name in enumerate(months, 1)}
        df_long['date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'].map(month_map).astype(str), errors='coerce')
        df_final = df_long[['date']].copy()
        df_final['value'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')
        df_final['group'] = '지구 평균 온도 이상치(℃)'
        return df_final.dropna(subset=['date', 'value'])
    except Exception as e:
        st.sidebar.error(f"GISTEMP 데이터 파싱 중 오류: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment() -> Optional[pd.DataFrame]:
    """Fetch World Bank API for Employment in industry, including ISO codes."""
    params = {'format': 'json', 'per_page': '20000'}
    resp = retry_get(CONFIG["worldbank_api_url"], params=params, max_retries=1)
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
        st.sidebar.error(f"World Bank 데이터 처리 중 오류: {e}")
        return None

def get_sample_climate_data() -> pd.DataFrame:
    """Generate sample climate data as a fallback."""
    years = pd.date_range(start=f"{TODAY.year-14}-01-01", end=f"{TODAY.year}-01-01", freq='MS')
    values = np.round(np.linspace(0.4, 1.1, len(years)) + np.random.normal(0, 0.05, len(years)), 3)
    return pd.DataFrame({'date': years, 'value': values, 'group': '지구 평균 온도 이상치(℃)'})

def get_sample_employment_data() -> pd.DataFrame:
    """Generate sample employment data as a fallback."""
    years = pd.date_range(start=f"{TODAY.year-9}-01-01", end=f"{TODAY.year}-01-01", freq='AS')
    data = []
    countries = {'한국(예시)': 'KOR', 'OECD 평균(예시)': 'OED'}
    for country, code in countries.items():
        base_value = 24.0 if '한국' in country else 22.0
        for year in years:
            data.append({'date': year, 'group': country, 'iso_code': code, 'value': float(base_value + np.random.normal(0, 0.8))})
    return pd.DataFrame(data)

# ==============================================================================
# 3. UI RENDERING FUNCTIONS
# ==============================================================================
def display_public_data_tab(climate_df: pd.DataFrame, employment_df: pd.DataFrame):
    """Render the content for the public data dashboard tab."""
    st.header("📈 공식 공개 데이터 기반 분석")
    st.markdown("NASA GISTEMP (기후)와 World Bank (고용)의 공개 데이터를 분석합니다. API 호출 실패 시 예시 데이터로 자동 대체됩니다.")

    # --- Sidebar controls for this tab ---
    st.sidebar.header("공개 데이터 옵션")
    show_trendline = st.sidebar.checkbox("🌡️ 5년 이동평균 추세선 표시", value=True, help="온도 데이터의 장기적 추세를 확인합니다.")
    
    # --- Key Metrics ---
    try:
        latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
        col1, col2 = st.columns(2)
        col1.metric(f"가장 최근 지구 온도 이상치 ({latest_climate['date'].strftime('%Y년 %m월')})", f"{latest_climate['value']:.2f} ℃", help="1951-1980년 평균 대비 온도 차이입니다.")
        col2.metric("고용 데이터 국가 수", f"{employment_df['group'].nunique()} 개", help="World Bank API에서 불러온 최신 데이터 기준입니다.")
    except (IndexError, ValueError):
        st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    st.markdown("---")

    # --- Climate Change Chart ---
    st.subheader("🌡️ 지구 평균 온도 이상치 변화")
    if not climate_df.empty:
        fig_climate = go.Figure()
        fig_climate.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['value'], mode='lines', name='월별 이상치', line=dict(width=1, color='lightblue')))
        if show_trendline:
            climate_df['trend'] = climate_df['value'].rolling(window=60, min_periods=12).mean()
            fig_climate.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['trend'], mode='lines', name='5년 이동평균', line=dict(width=3, color='royalblue')))
        st.plotly_chart(fig_climate, use_container_width=True)

    st.markdown("---")

    # --- Employment Data Section ---
    st.subheader("🏭 산업별 고용 비율 변화")
    if not employment_df.empty:
        employment_df['year'] = employment_df['date'].dt.year
        latest_year = int(employment_df['year'].max())
        
        # --- Choropleth Map ---
        st.markdown(f"**{latest_year}년 기준 전 세계 산업 고용 비율**")
        latest_year_df = employment_df[employment_df['year'] == latest_year]
        fig_map = px.choropleth(latest_year_df, locations="iso_code", color="value", hover_name="group",
                                color_continuous_scale=px.colors.sequential.Plasma,
                                labels={'value': '고용 비율 (%)'})
        st.plotly_chart(fig_map, use_container_width=True)

        # --- Country Comparison Chart ---
        st.markdown("**국가별 산업 고용 비율 추이 비교**")
        all_countries = sorted(employment_df['group'].unique())
        default_countries = [c for c in ['World', 'Korea, Rep.', 'China', 'United States', 'Germany'] if c in all_countries]
        if not default_countries and all_countries:
            default_countries = all_countries[:3]
        
        selected_countries = st.multiselect("비교할 국가를 선택하세요:", all_countries, default=default_countries)
        if selected_countries:
            comp_df = employment_df[employment_df['group'].isin(selected_countries)]
            fig_comp = px.line(comp_df, x='year', y='value', color='group', labels={'year':'연도', 'value':'산업 고용 비율(%)', 'group':'국가'})
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # --- Climate vs Employment Correlation ---
    st.subheader("🔄 기후(온도 이상치) vs 산업 고용(연 단위 비교)")
    normalize = st.checkbox("데이터 정규화 (스케일 맞춤)", help="단위가 다른 두 데이터를 0~1 사이 값으로 변환하여 추세 비교를 용이하게 합니다.")
    try:
        c_ann = climate_df.copy()
        c_ann['year'] = c_ann['date'].dt.year
        c_ann_agg = c_ann.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})
        e_ann = employment_df.copy()
        e_ann['year'] = e_ann['date'].dt.year
        e_ann_agg = e_ann.groupby('year')['value'].median().reset_index().rename(columns={'value':'industry_employment_median'})
        merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')

        if not merged.empty:
            if normalize:
                scaler = MinMaxScaler()
                merged[['temp_anomaly', 'industry_employment_median']] = scaler.fit_transform(merged[['temp_anomaly', 'industry_employment_median']])
            
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=merged['year'], y=merged['temp_anomaly'], name='기후 이상치(연평균)', yaxis='y1'))
            fig_corr.add_trace(go.Scatter(x=merged['year'], y=merged['industry_employment_median'], name='산업 고용(전세계 중앙값)', yaxis='y2'))
            fig_corr.update_layout(title_text="연도별 기후 이상치와 산업 고용 비율 비교", yaxis=dict(title="기후 이상치 (정규화)" if normalize else "기후 이상치(℃)"), yaxis2=dict(title="산업 고용 비율 (정규화)" if normalize else "산업 고용 비율(%)", overlaying='y', side='right'))
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"비교 그래프 생성 중 오류 발생: {e}")

def display_user_prompt_tab():
    """Render the content for the user prompt simulation tab."""
    st.header("📄 가상 시나리오 분석")
    st.markdown("외부 리포트의 **'녹색 경제 전환으로 2030년까지 녹색 일자리는 2,600만 개 증가하고, 화석 연료 기반 일자리는 1,500만 개 감소할 것이다'** 와 같은 문장을 가정하여 생성한 가상 데이터입니다.")
    
    years = list(range(2018, min(TODAY.year, 2031)))
    dates = [datetime.date(y, 1, 1) for y in years]
    user_jobs_df = pd.DataFrame({
        'date': dates * 2,
        'group': ['녹색 일자리(만 개)'] * len(years) + ['화석연료 일자리(만 개)'] * len(years),
        'value': np.linspace(5, 260, len(years)).tolist() + np.linspace(0, -150, len(years)).tolist()
    })
    user_jobs_df['date'] = pd.to_datetime(user_jobs_df['date'])
    
    st.sidebar.header("가상 데이터 옵션")
    min_year, max_year = user_jobs_df['date'].dt.year.min(), user_jobs_df['date'].dt.year.max()
    sel_start, sel_end = st.sidebar.slider("시뮬레이션 기간 선택", min_year, max_year, (min_year, max_year), key="user_date_slider")
    
    uj_filtered = user_jobs_df[(user_jobs_df['date'].dt.year >= sel_start) & (user_jobs_df['date'].dt.year <= sel_end)]

    st.subheader("💼 녹색 전환에 따른 일자리 변화 시뮬레이션")
    fig_u1 = px.line(uj_filtered, x='date', y='value', color='group', labels={'date':'연도', 'value':'일자리 변화(만 개)', 'group':'구분'}, markers=True)
    st.plotly_chart(fig_u1, use_container_width=True)

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    """Main function to run the Streamlit app."""
    st.title("기후 변화와 취업 동향 대시보드")

    if 'data_loaded' not in st.session_state:
        st.sidebar.title("데이터 로드 상태")
        with st.spinner("공식 공개 데이터를 불러오는 중입니다..."):
            climate_raw = fetch_gistemp_csv()
            emp_raw = fetch_worldbank_employment()
            st.session_state.climate_df = preprocess_dataframe(climate_raw if climate_raw is not None else get_sample_climate_data())
            st.session_state.employment_df = preprocess_dataframe(emp_raw if emp_raw is not None else get_sample_employment_data())
            if climate_raw is None: st.sidebar.error("NASA GISTEMP 로드 실패 → 예시 데이터 사용")
            else: st.sidebar.success("NASA GISTEMP 로드 성공")
            if emp_raw is None: st.sidebar.error("World Bank 데이터 로드 실패 → 예시 데이터 사용")
            else: st.sidebar.success("World Bank 데이터 로드 성공")
            st.session_state.data_loaded = True
            time.sleep(1)
            st.rerun() 
    
    tab1, tab2 = st.tabs(["🌏 공개 데이터 대시보드", " simulate 가상 시나리오"])
    with tab1:
        display_public_data_tab(st.session_state.climate_df, st.session_state.employment_df)
    with tab2:
        display_user_prompt_tab()

    with st.expander("개발자 및 실행 환경 참고사항"):
        st.markdown("""
        - 이 앱은 NASA/WorldBank 공개 API를 우선적으로 호출하며, 네트워크 실패 시 내장된 예시 데이터로 자동 전환됩니다.
        - **Kaggle 데이터 연동 방법**:
          1. `pip install kaggle`
          2. Kaggle 계정 > Settings > API > `Create New Token` 클릭하여 `kaggle.json` 다운로드
          3. 로컬 환경의 `~/.kaggle/kaggle.json` 위치에 저장 (`chmod 600 ~/.kaggle/kaggle.json`)
        """)

if __name__ == "__main__":
    try:
        if os.path.exists(CONFIG["font_path"]):
            st.markdown(f"""
            <style>
            @font-face {{ font-family: 'PretendardCustom'; src: url('{CONFIG["font_path"]}') format('truetype'); }}
            html, body, [class*="css"] {{ font-family: 'PretendardCustom', Pretendard, sans-serif; }}
            </style>""", unsafe_allow_html=True)
    except Exception: pass
    main()

