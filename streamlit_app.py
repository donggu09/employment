"""
Streamlit Dashboard (Korean) - V10.2 (Timeout Fix)
This version addresses potential `ConnectTimeoutError` issues by increasing the request timeout from 15 to 30 seconds. This makes the application more resilient to slow server responses, particularly from the NASA GISTEMP data source.
- Topic: 'The Impact of Climate Change on Employment'
- Core Features:
  1) Live public data dashboards via API calls with guaranteed fallbacks.
  2) In-depth analysis tab with correlation and job scenario simulator.
  3) A "Job Impact" section comparing green vs. at-risk jobs.
- UI/UX Enhancements:
  - **V10.2 Definitive Fix**:
    - **Increased Timeout**: Modified the `retry_get` function to use a 30-second timeout, reducing the likelihood of connection errors with slow-responding APIs.
    - **Corrected CO2 Parser**: Retained the fix for the NOAA CO2 data parser.
    - **Expanded Sample Data**: Retained the multi-year sample data.
    - **Robust Networking**: Retained the professional-grade requests.Session with a Retry adapter.
"""

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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# 0. CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="기후 변화와 미래 커리어 대시보드",
    page_icon="🌍",
    layout="wide"
)

# --- App constants ---
TODAY = datetime.datetime.now().date()
CONFIG = {
    "nasa_gistemp_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "worldbank_api_url": "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS",
    "noaa_co2_url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
}

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
    """
    Robust GET request using the global session with a retry adapter.
    Upon final failure, it logs the error to the session state for UI display.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; StreamlitApp/1.0)'}
    try:
        # [FIXED] Increased timeout from 15 to 30 seconds to handle slow server responses.
        resp = _SESSION.get(url, params=params, headers=headers, timeout=kwargs.get('timeout', 30), allow_redirects=True, verify=True)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        # Reformat the error message to be more user-friendly
        error_message = f"**API(`{url.split('//')[1].split('/')[0]}`) 요청 실패:** {e}"
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        if error_message not in st.session_state.api_errors:
            st.session_state.api_errors.append(error_message)
        return None

@st.cache_data(ttl=3600)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    # Ensure 'date' column exists and convert it to datetime
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date'])
        d = d[d['date'].dt.date <= TODAY]
    else:
        return pd.DataFrame() # Return empty if no date column

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
        if data_start_index == -1: return None
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
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        st.session_state.api_errors.append(f"**NASA GISTEMP 데이터 파싱 오류:** `{e}`")
        return None

@st.cache_data(ttl=3600)
def fetch_noaa_co2_data() -> Optional[pd.DataFrame]:
    resp = retry_get(CONFIG["noaa_co2_url"])
    if resp is None: return None
    try:
        column_names = [
            'year', 'month', 'decimal_date', 'average', 'interpolated',
            'trend', 'days', 'uncertainty'
        ]
        df = pd.read_csv(
            io.StringIO(resp.content.decode('utf-8')),
            comment='#',
            delim_whitespace=True,
            header=None,
            names=column_names
        )
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        df_final = df[['date', 'interpolated']].rename(columns={'interpolated': 'value'})
        df_final['group'] = '대기 중 CO₂ 농도 (ppm)'
        return df_final[df_final['value'] > 0]
    except Exception as e:
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        st.session_state.api_errors.append(f"**NOAA CO₂ 데이터 파싱 오류:** `{e}`")
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
        if 'api_errors' not in st.session_state:
            st.session_state.api_errors = []
        st.session_state.api_errors.append(f"**World Bank 데이터 파싱 오류:** `{e}`")
        return None

# --- [EXPANDED] Embedded Sample Data Fallbacks ---
@st.cache_data
def get_sample_climate_data() -> pd.DataFrame:
    csv_data = """date,value,group
2020-01-01,1.16,"지구 평균 온도 이상치(℃) (예시)"
2020-07-01,0.92,"지구 평균 온도 이상치(℃) (예시)"
2021-01-01,0.86,"지구 평균 온도 이상치(℃) (예시)"
2021-07-01,0.92,"지구 평균 온도 이상치(℃) (예시)"
2022-01-01,0.91,"지구 평균 온도 이상치(℃) (예시)"
2022-07-01,0.94,"지구 평균 온도 이상치(℃) (예시)"
2023-01-01,1.08,"지구 평균 온도 이상치(℃) (예시)"
2023-07-01,1.24,"지구 평균 온도 이상치(℃) (예시)"
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_data
def get_sample_co2_data() -> pd.DataFrame:
    csv_data = """date,value,group
2020-01-01,413.4,"대기 중 CO₂ 농도 (ppm) (예시)"
2020-07-01,414.72,"대기 중 CO₂ 농도 (ppm) (예시)"
2021-01-01,415.4,"대기 중 CO₂ 농도 (ppm) (예시)"
2021-07-01,416.96,"대기 중 CO₂ 농도 (ppm) (예시)"
2022-01-01,418.28,"대기 중 CO₂ 농도 (ppm) (예시)"
2022-07-01,418.91,"대기 중 CO₂ 농도 (ppm) (예시)"
2023-01-01,420.51,"대기 중 CO₂ 농도 (ppm) (예시)"
2023-07-01,421.84,"대기 중 CO₂ 농도 (ppm) (예시)"
"""
    return pd.read_csv(io.StringIO(csv_data))

@st.cache_data
def get_sample_employment_data() -> pd.DataFrame:
    csv_data = """date,group,iso_code,value
2020-01-01,World (예시),WLD,20.53
2021-01-01,World (예시),WLD,20.81
2022-01-01,World (예시),WLD,21.0
2023-01-01,World (예시),WLD,21.2
2020-01-01,Korea (예시),KOR,23.2
2021-01-01,Korea (예시),KOR,23.5
2022-01-01,Korea (예시),KOR,23.7
2023-01-01,Korea (예시),KOR,23.9
"""
    return pd.read_csv(io.StringIO(csv_data))

# ==============================================================================
# 3. UI RENDERING FUNCTIONS FOR TABS
# ==============================================================================
# --------------------------- Data Status UI -----------------------------
def display_data_status():
    st.subheader("데이터 출처 현황")
    status = st.session_state.get('data_status', {})
    
    cols = st.columns(3)
    
    nasa_status = status.get('climate', 'N/A')
    noaa_status = status.get('co2', 'N/A')
    wb_status = status.get('employment', 'N/A')

    with cols[0]:
        st.markdown(f"**NASA GISTEMP (기온)**: { '🟢 실시간' if nasa_status == 'Live' else '🟡 예시'}")
    with cols[1]:
        st.markdown(f"**NOAA CO₂ (이산화탄소)**: { '🟢 실시간' if noaa_status == 'Live' else '🟡 예시'}")
    with cols[2]:
        st.markdown(f"**World Bank (고용)**: { '🟢 실시간' if wb_status == 'Live' else '🟡 예시'}")
    st.markdown("---")

# --------------------------- API Error UI -----------------------------
def display_api_errors():
    """Displays any API errors that were collected during the data loading process."""
    if st.session_state.get('api_errors'):
        st.subheader("⚠️ API 호출 또는 데이터 처리 오류")
        for error in st.session_state.api_errors:
            st.error(error, icon="🔥")
        st.markdown("---")


# --------------------------- TAB 1: Global Trends -----------------------------
def display_global_trends_tab(climate_df, co2_df, employment_df):
    st.header("📈 글로벌 기후 및 고용 동향")
    st.markdown("NASA, NOAA, World Bank의 최신 데이터를 시각화합니다.")
    
    col1, col2, col3 = st.columns(3)
    if not climate_df.empty and not co2_df.empty and not employment_df.empty:
        try:
            # Ensure date column is datetime before formatting
            climate_df['date'] = pd.to_datetime(climate_df['date'])
            co2_df['date'] = pd.to_datetime(co2_df['date'])

            latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
            latest_co2 = co2_df.sort_values('date', ascending=False).iloc[0]
            col1.metric(f"최신 온도 이상치 ({latest_climate['date']:%Y-%m})", f"{latest_climate['value']:.2f} ℃")
            col2.metric(f"최신 CO₂ 농도 ({latest_co2['date']:%Y-%m})", f"{latest_co2['value']:.2f} ppm")
            col3.metric("고용 데이터 국가 수", f"{employment_df['group'].nunique()} 개")
        except (IndexError, ValueError, TypeError): 
            st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    else:
        st.info("데이터를 불러오는 중이거나 API 호출에 실패하여 표시할 데이터가 없습니다.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌡️ 지구 평균 온도 이상치")
        if not climate_df.empty:
            fig = px.line(climate_df, x='date', y='value', labels={'date': '', 'value': '온도 이상치 (°C)'}, color_discrete_sequence=['#d62728'])
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("💨 대기 중 CO₂ 농도 (마우나로아)")
        if not co2_df.empty:
            fig = px.line(co2_df, x='date', y='value', labels={'date': '', 'value': 'CO₂ (ppm)'}, color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("🏭 산업별 고용 비율 변화")
    if not employment_df.empty:
        employment_df['year'] = pd.to_datetime(employment_df['date']).dt.year
        min_year, max_year = int(employment_df['year'].min()), int(employment_df['year'].max())
        selected_year = st.slider("연도 선택:", min_year, max_year, max_year, key="map_year_slider")
        
        map_df = employment_df[employment_df['year'] == selected_year]
        if not map_df.empty:
            fig_map = px.choropleth(map_df, locations="iso_code", color="value", hover_name="group", color_continuous_scale=px.colors.sequential.Plasma, labels={'value': '고용 비율 (%)'}, title=f"{selected_year}년 전 세계 산업 고용 비율")
            st.plotly_chart(fig_map, use_container_width=True)

        all_countries = sorted(employment_df['group'].unique())
        default_countries = [c for c in ['World', 'Korea, Rep.', 'World (예시)', 'Korea (예시)'] if c in all_countries] or all_countries[:2]
        selected_countries = st.multiselect("국가별 추이 비교:", all_countries, default=default_countries)
        if selected_countries:
            comp_df = employment_df[employment_df['group'].isin(selected_countries)]
            fig_comp = px.line(comp_df, x='year', y='value', color='group', labels={'year':'연도', 'value':'산업 고용 비율(%)', 'group':'국가'})
            st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------- TAB 2: In-Depth Analysis ---------------------------
def display_analysis_tab(climate_df, co2_df, employment_df):
    st.header("🔍 심층 분석: 상관관계와 미래 시뮬레이션")
    
    with st.container(border=True):
        st.subheader("🔄 기후 지표 vs. 산업 고용 상관관계")
        if climate_df.empty or co2_df.empty or employment_df.empty:
            st.warning("상관관계를 분석하기 위한 데이터가 부족합니다.")
            return

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
                return
            
            corr_col1, corr_col2 = st.columns(2)
            corr_choice = corr_col1.selectbox("비교할 기후 지표:", ('온도 이상치', 'CO₂ 농도'))
            normalize = corr_col2.checkbox("데이터 정규화 (0-1 스케일)", help="단위가 다른 두 데이터를 0~1 사이 값으로 변환하여 추세 비교를 용이하게 합니다.")
            
            x_var = 'temp_anomaly' if corr_choice == '온도 이상치' else 'co2_ppm'
            y_var = 'employment_median'
            
            plot_df = merged[['year', x_var, y_var]].copy()
            correlation = plot_df[x_var].corr(plot_df[y_var])
            st.metric(f"{corr_choice} vs. 고용 비율 상관계수", f"{correlation:.3f}")

            if normalize:
                plot_df[x_var] = (plot_df[x_var] - plot_df[x_var].min()) / (plot_df[x_var].max() - plot_df[x_var].min())
                plot_df[y_var] = (plot_df[y_var] - plot_df[y_var].min()) / (plot_df[y_var].max() - plot_df[y_var].min())
            
            # Create a figure with a secondary y-axis
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[x_var], name=corr_choice,
                                          line=dict(color='#d62728')))
            fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[y_var], name='산업 고용(전세계 중앙값)', yaxis='y2',
                                          line=dict(color='#1f77b4')))

            # Update layout for the secondary y-axis
            fig_corr.update_layout(
                xaxis_title="연도",
                yaxis_title=f"{corr_choice} ({'℃' if x_var == 'temp_anomaly' else 'ppm'})" if not normalize else "정규화된 값",
                yaxis2=dict(
                    title="산업 고용 비율 (%)" if not normalize else "정규화된 값",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")

    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("📄 가상 시나리오 분석")
        col1, col2 = st.columns(2)
        green_growth_rate = col1.slider("연간 녹색 일자리 성장률 (%)", 1.0, 20.0, 10.0, 0.5, key="sim_growth") / 100
        fossil_decline_rate = col2.slider("연간 화석연료 일자리 감소율 (%)", 1.0, 20.0, 8.0, 0.5, key="sim_decline") / 100
        
        years = list(range(2025, 2041))
        green_jobs, fossil_jobs = [500], [1000]
        for _ in range(1, len(years)):
            green_jobs.append(green_jobs[-1] * (1 + green_growth_rate))
            fossil_jobs.append(fossil_jobs[-1] * (1 - fossil_decline_rate))

        user_jobs_df = pd.DataFrame({ 'date': pd.to_datetime([datetime.date(y, 1, 1) for y in years] * 2), 'group': ['녹색 일자리(만 개)'] * len(years) + ['화석연료 일자리(만 개)'] * len(years), 'value': green_jobs + fossil_jobs })
        fig = px.line(user_jobs_df, x='date', y='value', color='group', color_discrete_map={'녹색 일자리(만 개)': '#2ca02c', '화석연료 일자리(만 개)': '#7f7f7f'})
        st.plotly_chart(fig, use_container_width=True)

# --------------------------- TAB 3: Job Impact --------------------------------
def display_job_impact_tab():
    st.header("⚖️ 녹색 전환: 기회와 위험 직무 비교")
    df_op = pd.DataFrame({ '직무': ['기후 데이터 분석가', '탄소배출권 전문가', '신재생 에너지 개발자', 'ESG 컨설턴트', '스마트팜 전문가'], '성장 가능성 (점수)': [95, 90, 88, 85, 82] })
    df_r = pd.DataFrame({ '직무': ['화력 발전소 기술자', '자동차 내연기관 엔지니어', '석유화학 공장 운영원', '벌목업 종사자'], '위험도 (점수)': [90, 85, 80, 75] })
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 성장 가능성이 높은 녹색 직무")
        fig_op = px.bar(df_op, x='성장 가능성 (점수)', y='직무', orientation='h', color='성장 가능성 (점수)', color_continuous_scale=px.colors.sequential.Greens)
        st.plotly_chart(fig_op, use_container_width=True)
    with col2:
        st.subheader("⚠️ 전환이 필요한 기존 직무")
        fig_risk = px.bar(df_r, x='위험도 (점수)', y='직무', orientation='h', color='위험도 (점수)', color_continuous_scale=px.colors.sequential.Reds)
        st.plotly_chart(fig_risk, use_container_width=True)


# ----------------------- TAB 4: Career Simulation Game ------------------------
def display_career_game_tab():
    st.header("🚀 나의 미래 설계하기 (커리어 시뮬레이션)")
    st.info("당신의 선택이 10년 후 커리어와 환경에 어떤 영향을 미치는지 시뮬레이션 해보세요!")

    with st.form("career_game_form"):
        # --- Stage 1: University ---
        with st.expander("🎓 1단계: 대학생", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                major = st.radio("주요 전공을 선택하세요:",
                                 ("컴퓨터공학 (AI 트랙)", "기계공학", "경제학"), key="major")
            with col2:
                club = st.radio("핵심 동아리 활동은 무엇인가요?",
                                ("신재생에너지 정책 토론", "코딩 스터디", "문학 비평"), key="club")
            with col3:
                project = st.radio("졸업 프로젝트 주제는 무엇인가요?",
                                   ("탄소 배출량 예측 AI 모델", "고효율 내연기관 설계", "ESG 경영사례 분석"), key="project")

        # --- Stage 2: Early Career ---
        with st.expander("💼 2단계: 사회초년생", expanded=True):
            col4, col5, col6 = st.columns(3)
            with col4:
                first_job = st.radio("첫 직장을 선택하세요:",
                                     ("에너지 IT 스타트업", "대기업 정유회사", "금융권 애널리스트"), key="first_job")
            with col5:
                skill_dev = st.radio("어떤 역량을 집중적으로 키울 건가요?",
                                     ("클라우드 기반 데이터 분석", "전통 공정 관리", "재무 분석 및 투자"), key="skill_dev")
            with col6:
                side_project = st.radio("개인적으로 진행할 프로젝트는?",
                                        ("오픈소스 기후 데이터 시각화", "자동차 연비 개선 연구", "주식 투자 포트폴리오 관리"), key="side_project")
        
        submitted = st.form_submit_button("🚀 나의 미래 확인하기")

    if submitted:
        # --- Scoring Logic ---
        career_score, green_score = 0, 0
        skills = {"데이터 분석":0, "정책/경영":0, "엔지니어링":0, "금융/경제":0}

        # Stage 1 Scoring
        if major == "컴퓨터공학 (AI 트랙)": career_score += 20; green_score += 10; skills["데이터 분석"] += 2
        elif major == "기계공학": career_score += 10; green_score += 0; skills["엔지니어링"] += 2
        else: career_score += 15; green_score += 5; skills["금융/경제"] += 2

        if club == "신재생에너지 정책 토론": career_score += 10; green_score += 15; skills["정책/경영"] += 1
        elif club == "코딩 스터디": career_score += 15; green_score += 5; skills["데이터 분석"] += 1
        else: career_score += 5; green_score += 0

        if project == "탄소 배출량 예측 AI 모델": career_score += 15; green_score += 20; skills["데이터 분석"] += 1; skills["정책/경영"] += 1
        elif project == "고효율 내연기관 설계": career_score += 5; green_score -= 10; skills["엔지니어링"] += 1
        else: career_score += 10; green_score += 10; skills["정책/경영"] += 1; skills["금융/경제"] += 1

        # Stage 2 Scoring
        if first_job == "에너지 IT 스타트업": career_score += 15; green_score += 20
        elif first_job == "대기업 정유회사": career_score += 20; green_score -= 10
        else: career_score += 15; green_score += 5

        if skill_dev == "클라우드 기반 데이터 분석": career_score += 20; green_score += 10; skills["데이터 분석"] += 2
        elif skill_dev == "전통 공정 관리": career_score += 10; green_score -= 5; skills["엔지니어링"] += 1
        else: career_score += 15; green_score += 0; skills["금융/경제"] += 1

        if side_project == "오픈소스 기후 데이터 시각화": career_score += 10; green_score += 15; skills["데이터 분석"] += 1
        elif side_project == "자동차 연비 개선 연구": career_score += 5; green_score -= 5; skills["엔지니어링"] += 1
        else: career_score += 5; green_score += 0; skills["금융/경제"] += 1
        
        # --- Determine Job Title ---
        if green_score >= 50 and career_score >= 70: job_title = "기후 기술 최고 전문가"
        elif green_score >= 30 and career_score >= 60: job_title = "그린 에너지 전략가"
        elif career_score >= 70: job_title = "산업 전문가"
        elif green_score >= 30: job_title = "환경 정책가"
        else: job_title = "미래 준비형 인재"

        st.subheader("🎉 최종 결과: 당신의 커리어 카드")
        with st.container(border=True):
            res1, res2 = st.columns([0.6, 0.4])
            with res1:
                st.markdown(f"#### 💼 직업: {job_title}")
                st.metric("🚀 미래 전망 점수", f"{career_score} / 100")
                st.metric("🌱 환경 기여도 점수", f"{green_score} / 75")

            with res2:
                df_skills = pd.DataFrame(dict(r=list(skills.values()), theta=list(skills.keys())))
                fig = px.line_polar(df_skills, r='r', theta='theta', line_close=True, range_r=[0,5])
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("나의 역량 레이더 차트")

# ----------------------- TAB 5: Survey & Feedback ------------------------
def display_survey_tab():
    st.header("📝 설문 및 의견")
    st.markdown("기후 변화와 미래 직업에 대한 여러분의 소중한 의견을 들려주세요!")

    with st.form("survey_form"):
        st.subheader("개인 인식")
        q1 = st.radio("1️⃣ 기후변화가 나의 직업(또는 미래 직업)에 영향을 줄 것이라 생각하시나요?", ["매우 그렇다", "조금 그렇다", "별로 아니다", "전혀 아니다"])
        q2 = st.slider("2️⃣ 기후변화 대응 역량을 키우고 싶은 정도는 어느 정도인가요? (0~10점)", 0, 10, 5)
        
        st.subheader("직업 선호도")
        q3 = st.selectbox("3️⃣ 가장 관심 있는 녹색 일자리 분야는 무엇인가요?", ["신재생에너지", "ESG 컨설팅", "탄소 배출권 거래", "기후 데이터 분석", "스마트팜/친환경 농업", "기타"])
        q4 = st.multiselect("4️⃣ 녹색 일자리 전환 시 가장 필요하다고 생각하는 지원은 무엇인가요? (중복 선택 가능)", ["전문 재교육 프로그램", "정부의 재정 지원", "기업의 채용 연계", "멘토링 및 상담"])
        
        st.subheader("정책 및 사회")
        q5 = st.radio("5️⃣ 기후변화 대응을 위해 세금(탄소세 등)을 더 내는 것에 동의하시나요?", ["적극 찬성", "찬성", "반대", "적극 반대"])
        q6 = st.text_area("6️⃣ 녹색 일자리 확대를 위해 가장 필요하다고 생각하는 정책이나 제안이 있다면 자유롭게 적어주세요.")
        
        submitted = st.form_submit_button("설문 제출하기")

    if submitted:
        st.success("✅ 설문에 참여해주셔서 감사합니다! 아래는 나의 응답 요약입니다.")
        
        with st.container(border=True):
            st.subheader("📋 나의 응답 요약")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**기후변화 영향 인식:** {q1}")
                st.write(f"**관심 녹색 일자리:** {q3}")
                st.write(f"**필요한 지원:** {', '.join(q4) if q4 else '선택 안함'}")
                st.write(f"**탄소세 동의:** {q5}")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = q2,
                    title = {'text': "나의 역량 개발 의지 점수"},
                    gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "#2ca02c"}}))
                st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    # [FIXED] Updated title to match version V10.2
    st.title("기후 변화와 미래 커리어 대시보드 V10.2 (타임아웃 수정) 🌍💼")

    # --- Data Loading ---
    if 'data_loaded' not in st.session_state:
        st.session_state.data_status = {}
        st.session_state.api_errors = [] # Initialize error list

        with st.spinner("실시간 데이터를 불러오는 중입니다..."):
            
            # --- Unified Data Pipeline ---
            
            # Climate Data
            climate_raw = fetch_gistemp_csv()
            if climate_raw is not None and not climate_raw.empty:
                st.session_state.data_status['climate'] = 'Live'
            else:
                climate_raw = get_sample_climate_data()
                st.session_state.data_status['climate'] = 'Sample'
            st.session_state.climate_df = preprocess_dataframe(climate_raw)

            # CO2 Data
            co2_raw = fetch_noaa_co2_data()
            if co2_raw is not None and not co2_raw.empty:
                st.session_state.data_status['co2'] = 'Live'
            else:
                co2_raw = get_sample_co2_data()
                st.session_state.data_status['co2'] = 'Sample'
            st.session_state.co2_df = preprocess_dataframe(co2_raw)

            # Employment Data
            wb_employment_raw = fetch_worldbank_employment()
            if wb_employment_raw is not None and not wb_employment_raw.empty:
                st.session_state.data_status['employment'] = 'Live'
            else:
                wb_employment_raw = get_sample_employment_data()
                st.session_state.data_status['employment'] = 'Sample'
            st.session_state.employment_df = preprocess_dataframe(wb_employment_raw)
            
            st.session_state.data_loaded = True
            time.sleep(0.5)
            st.rerun()
    
    # --- Display Status and Error Panels ---
    display_data_status()
    display_api_errors()
    
    # --- Tabbed Interface ---
    tabs = st.tabs(["📊 글로벌 동향", "🔍 심층 분석", "⚖️ 직무 영향 분석", "🚀 나의 미래 설계하기", "📝 설문 및 의견"])
    with tabs[0]:
        display_global_trends_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df)
    with tabs[1]:
        display_analysis_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df)
    with tabs[2]:
        display_job_impact_tab()
    with tabs[3]:
        display_career_game_tab()
    with tabs[4]:
        display_survey_tab()

if __name__ == "__main__":
    main()

