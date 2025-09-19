"""
Streamlit Dashboard (Korean) - FULLY INTEGRATED VERSION (V2)

Topic: 'The Impact of Climate Change on Employment'

Merged Features:
- Advanced interactive dashboard with real-time public data (NASA, NOAA, World Bank).
- Interactive 'what-if' scenario simulation for job market forecasting.
- A detailed, text-based 'Analysis Report' page for qualitative insights.
- Sidebar navigation to switch between the features.

V2 Enhancements:
- Added 'Promising Future Jobs' page with career-focused information.
- Added renewable energy generation data visualization to the main dashboard.
- Improved UI/UX with a card-based layout for better readability and aesthetics.
- Added icons to headers for intuitive navigation.

Data Robustness:
- All API calls have fallbacks to generate sample data upon failure.
- Data is cached and stored in session_state for efficient performance.
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

# ==============================================================================
# 0. CONFIGURATION & INITIAL SETUP
# ==============================================================================
st.set_page_config(
    page_title="기후와 취업: 통합 대시보드",
    page_icon="🌍",
    layout="wide"
)

# --- App constants ---
TODAY = datetime.datetime.now().date()
CONFIG = {
    "nasa_gistemp_url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "worldbank_api_url": "https://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS",
    "noaa_co2_url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
    "font_path": "/fonts/Pretendard-Bold.ttf",
}

# ==============================================================================
# 1. UTILITY & DATA LOADING FUNCTIONS
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

def normalize_series(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to a 0-1 scale."""
    if s.max() == s.min(): return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

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
def fetch_noaa_co2_data() -> Optional[pd.DataFrame]:
    """Fetch and parse NOAA Mauna Loa CO2 data."""
    resp = retry_get(CONFIG["noaa_co2_url"], max_retries=1)
    if resp is None: return None
    try:
        content = resp.content.decode('utf-8')
        lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
        df = pd.read_csv(io.StringIO('\n'.join(lines)), delim_whitespace=True, header=None,
                         names=['year', 'month', 'decimal_date', 'value', 'value_deseasonalized', 'num_days', 'stdev', 'unc'])
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df_final = df[['date', 'value']].copy()
        df_final['group'] = '대기 중 CO₂ 농도 (ppm)'
        return df_final[df_final['value'] > 0].reset_index(drop=True)
    except Exception as e:
        st.sidebar.error(f"NOAA CO2 데이터 파싱 중 오류: {e}")
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
    dates = pd.date_range(end=TODAY, periods=14*12, freq='MS')
    values = np.round(np.linspace(0.4, 1.2, len(dates)) + np.random.normal(0, 0.05, len(dates)), 3)
    return pd.DataFrame({'date': dates, 'value': values, 'group': '지구 평균 온도 이상치(℃)'})

def get_sample_co2_data() -> pd.DataFrame:
    """Generate sample CO2 data as a fallback."""
    dates = pd.date_range(end=TODAY, periods=14*12, freq='MS')
    values = np.round(np.linspace(380, 420, len(dates)) + np.random.normal(0, 0.5, len(dates)), 2)
    return pd.DataFrame({'date': dates, 'value': values, 'group': '대기 중 CO₂ 농도 (ppm)'})

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

@st.cache_data
def get_sample_renewable_data() -> pd.DataFrame:
    """Generate sample renewable energy data as a fallback."""
    years = list(range(2010, 2024))
    data = {
        '연도': years,
        '태양광 (TWh)': [30 + i**2.1 for i in range(len(years))],
        '풍력 (TWh)': [80 + i**2.3 for i in range(len(years))],
        '수력 (TWh)': [3400 + i*30 for i in range(len(years))],
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='연도', var_name='에너지원', value_name='발전량 (TWh)')
    return df_melted

# ==============================================================================
# 2. UI RENDERING FUNCTIONS
# ==============================================================================
def display_public_data_tab(climate_df: pd.DataFrame, co2_df: pd.DataFrame, employment_df: pd.DataFrame, renewable_df: pd.DataFrame):
    """Render the content for the public data dashboard tab."""
    st.header("📈 공식 공개 데이터 기반 분석")
    st.markdown("NASA (기온), NOAA (CO₂), World Bank (고용)의 공개 데이터를 분석합니다. API 호출 실패 시 예시 데이터로 자동 대체됩니다.")

    with st.container(border=True):
        st.subheader("📊 핵심 지표 요약", divider='rainbow')
        try:
            latest_climate = climate_df.sort_values('date', ascending=False).iloc[0]
            latest_co2 = co2_df.sort_values('date', ascending=False).iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric(f"최신 온도 이상치 ({latest_climate['date']:%Y-%m})", f"{latest_climate['value']:.2f} ℃")
            col2.metric(f"최신 CO₂ 농도 ({latest_co2['date']:%Y-%m})", f"{latest_co2['value']:.2f} ppm")
            col3.metric("고용 데이터 국가 수", f"{employment_df['group'].nunique()} 개")
        except (IndexError, ValueError):
            st.info("핵심 지표를 계산할 데이터가 부족합니다.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("🌡️ 지구 평균 온도 이상치")
            show_trendline = st.checkbox("5년 이동평균 추세선", value=True, key="trend_cb")
            if not climate_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['value'], mode='lines', name='월별 이상치', line=dict(width=1, color='lightblue')))
                if show_trendline:
                    climate_df['trend'] = climate_df['value'].rolling(window=60, min_periods=12).mean()
                    fig.add_trace(go.Scatter(x=climate_df['date'], y=climate_df['trend'], mode='lines', name='5년 이동평균', line=dict(width=3, color='royalblue')))
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("온도 데이터 다운로드", climate_df.to_csv(index=False, encoding='utf-8-sig'), "climate_data.csv", "text/csv", key="dl_climate")

    with c2:
        with st.container(border=True):
            st.subheader("💨 대기 중 CO₂ 농도")
            st.markdown("<p style='font-size: smaller;'>하와이 마우나로아 관측소 기준</p>", unsafe_allow_html=True)
            if not co2_df.empty:
                fig = px.line(co2_df, x='date', y='value', labels={'date': '날짜', 'value': 'CO₂ (ppm)'})
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("CO₂ 데이터 다운로드", co2_df.to_csv(index=False, encoding='utf-8-sig'), "co2_data.csv", "text/csv", key="dl_co2")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("🏭 산업별 고용 비율 변화")
        if not employment_df.empty:
            employment_df['year'] = employment_df['date'].dt.year
            min_year = int(employment_df['year'].min())
            max_year = int(employment_df['year'].max())
            selected_year = st.slider("연도를 선택하여 지도를 변경하세요:", min_year, max_year, max_year)
            st.markdown(f"**{selected_year}년 기준 전 세계 산업 고용 비율 (Choropleth Map)**")
            map_df = employment_df[employment_df['year'] == selected_year]
            if not map_df.empty:
                fig_map = px.choropleth(map_df, locations="iso_code", color="value", hover_name="group", color_continuous_scale=px.colors.sequential.Plasma, labels={'value': '고용 비율 (%)'})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning(f"{selected_year}년에는 표시할 고용 데이터가 없습니다.")

            st.markdown("**국가별 산업 고용 비율 추이 비교**")
            all_countries = sorted(employment_df['group'].unique())
            default_countries = [c for c in ['World', 'Korea, Rep.', 'China', 'United States', 'Germany'] if c in all_countries] or all_countries[:3]
            selected_countries = st.multiselect("비교할 국가를 선택하세요:", all_countries, default=default_countries)
            if selected_countries:
                comp_df = employment_df[employment_df['group'].isin(selected_countries)]
                fig_comp = px.line(comp_df, x='year', y='value', color='group', labels={'year':'연도', 'value':'산업 고용 비율(%)', 'group':'국가'})
                st.plotly_chart(fig_comp, use_container_width=True)
                st.download_button("선택 국가 고용 데이터 다운로드", comp_df.to_csv(index=False, encoding='utf-8-sig'), "employment_selected.csv", "text/csv", key="dl_emp")

    st.markdown("<br>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.subheader("🔄 기후 지표 vs. 산업 고용 상관관계")
            try:
                c_ann = climate_df.copy(); c_ann['year'] = c_ann['date'].dt.year
                c_ann_agg = c_ann.groupby('year')['value'].mean().reset_index().rename(columns={'value':'temp_anomaly'})
                co2_ann = co2_df.copy(); co2_ann['year'] = co2_ann['date'].dt.year
                co2_ann_agg = co2_ann.groupby('year')['value'].mean().reset_index().rename(columns={'value':'co2_ppm'})
                e_ann = employment_df.copy(); e_ann['year'] = e_ann['date'].dt.year
                e_ann_agg = e_ann.groupby('year')['value'].median().reset_index().rename(columns={'value':'employment_median'})
                merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')
                merged = pd.merge(merged, co2_ann_agg, on='year', how='inner')
                
                corr_choice = st.selectbox("고용 데이터와 비교할 기후 지표를 선택하세요:", ('온도 이상치', 'CO₂ 농도'))
                normalize = st.checkbox("데이터 정규화 (0-1 스케일)", help="단위가 다른 두 데이터를 0~1 사이 값으로 변환하여 추세 비교를 용이하게 합니다.")
                
                x_var = 'temp_anomaly' if corr_choice == '온도 이상치' else 'co2_ppm'
                y_var = 'employment_median'
                plot_df = merged[['year', x_var, y_var]].copy()
                correlation = plot_df[x_var].corr(plot_df[y_var])
                st.metric(f"{corr_choice} vs. 고용 비율 상관계수", f"{correlation:.3f}", help="Pearson 상관계수. 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 의미합니다.")
                
                if normalize:
                    plot_df[x_var] = normalize_series(plot_df[x_var])
                    plot_df[y_var] = normalize_series(plot_df[y_var])
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[x_var], name=corr_choice, yaxis='y1'))
                fig_corr.add_trace(go.Scatter(x=plot_df['year'], y=plot_df[y_var], name='산업 고용(전세계 중앙값)', yaxis='y2'))
                fig_corr.update_layout(title_text=f"연도별 {corr_choice}와 산업 고용 비율 비교", yaxis=dict(title=f"{corr_choice} (정규화)" if normalize else ('℃' if x_var=='temp_anomaly' else 'ppm')), yaxis2=dict(title="산업 고용 비율 (정규화)" if normalize else "%", overlaying='y', side='right'))
                st.plotly_chart(fig_corr, use_container_width=True)
                st.download_button("상관관계 데이터 다운로드", plot_df.to_csv(index=False, encoding='utf-8-sig'), "correlation_data.csv", "text/csv", key="dl_corr")
            except Exception as e:
                st.error(f"상관관계 분석 중 오류 발생: {e}")
    with c4:
        with st.container(border=True):
            st.subheader("☀️ 전 세계 재생에너지 발전량 (예시)")
            st.markdown("<p style='font-size: smaller;'>IEA 등 국제기구 데이터를 기반으로 한 예시 데이터입니다.</p>", unsafe_allow_html=True)
            fig = px.area(renewable_df, x='연도', y='발전량 (TWh)', color='에너지원', title='재생에너지원별 발전량 추이')
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("재생에너지 데이터 다운로드", renewable_df.to_csv(index=False, encoding='utf-8-sig'), "renewable_data.csv", "text/csv", key="dl_renewable")


def display_user_prompt_tab():
    """Render the content for the user prompt simulation tab."""
    st.header("📄 가상 시나리오 분석")
    st.markdown("외부 리포트의 예측을 바탕으로, **사용자가 직접 변수를 조절**하며 미래 일자리 변화를 시뮬레이션할 수 있습니다.")

    with st.container(border=True):
        st.subheader("⚙️ 시나리오 변수 조절", divider='rainbow')
        col1, col2 = st.columns(2)
        green_growth_rate = col1.slider("연간 녹색 일자리 성장률 (%)", 1.0, 20.0, 10.0, 0.5) / 100
        fossil_decline_rate = col2.slider("연간 화석연료 일자리 감소율 (%)", 1.0, 20.0, 8.0, 0.5) / 100
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        years = list(range(2025, 2042))
        green_jobs = [500]
        fossil_jobs = [1000]
        for _ in range(1, len(years)):
            green_jobs.append(green_jobs[-1] * (1 + green_growth_rate))
            fossil_jobs.append(fossil_jobs[-1] * (1 - fossil_decline_rate))
        user_jobs_df = pd.DataFrame({
            'date': pd.to_datetime([datetime.date(y, 1, 1) for y in years] * 2),
            'group': ['녹색 일자리(만 개)'] * len(years) + ['화석연료 일자리(만 개)'] * len(years),
            'value': green_jobs + fossil_jobs
        })
        st.subheader(f"💼 {years[0]}년 ~ {years[-1]}년 일자리 변화 시뮬레이션")
        fig = px.line(user_jobs_df, x='date', y='value', color='group', labels={'date':'연도', 'value':'총 일자리 수(만 개)', 'group':'구분'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**{years[-1]}년 예측 결과**")
        m1, m2, m3 = st.columns(3)
        final_green = user_jobs_df[user_jobs_df['group'] == '녹색 일자리(만 개)']['value'].iloc[-1]
        final_fossil = user_jobs_df[user_jobs_df['group'] == '화석연료 일자리(만 개)']['value'].iloc[-1]
        m1.metric("녹색 일자리", f"{final_green:,.0f} 만 개")
        m2.metric("화석연료 일자리", f"{final_fossil:,.0f} 만 개")
        m3.metric("총 일자리 변화", f"{((final_green + final_fossil) - (500 + 1000)):,.0f} 만 개", delta_color="off")
        st.download_button("시뮬레이션 결과 다운로드", user_jobs_df.to_csv(index=False, encoding='utf-8-sig'), "scenario_data.csv", "text/csv", key="dl_scenario")

def display_report_page():
    """Render the content for the static analysis report page."""
    st.title("📝 기후변화와 취업 보고서")
    
    with st.container(border=True):
        st.header("서론")
        st.write("""
        최근 기후위기는 단순한 환경 문제가 아니라 **청년의 미래 직업 선택**에도 큰 영향을 주고 있습니다.
        
        본 보고서는 기후변화가 산업 구조와 고용률에 어떤 변화를 일으키는지, 특히 청년 취업률에 미치는 영향을 분석하기 위해 작성되었습니다.
        """)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.header("본론")
        st.subheader("1. 기후 변화와 산업 구조 변화", divider='blue')
        st.write("""
        - NASA 기온 데이터에 따르면 지구 평균 기온은 꾸준히 상승하고 있습니다.
        - 이로 인해 **녹색산업 일자리**는 매년 증가하고 있으며, 반대로 **화석연료 관련 일자리**는 감소세를 보이고 있습니다.
        """)
        st.subheader("2. 청년 취업률 변화", divider='blue')
        st.write("""
        - 한국직업능력연구원(YPEC) 통계에 따르면 대졸 청년 취업률은 2017년 약 67%에서 2022년 68% 수준으로 등락을 반복하고 있습니다.
        - 그러나 **친환경 전공 취업률은 꾸준히 상승**하는 반면, **비친환경 전공 취업률은 하락세**를 보입니다.
        """)
        st.subheader("3. 정책과 미래 직업 방향", divider='blue')
        st.write("""
        - 고용노동부와 정부는 탄소중립 정책을 통해 녹색산업에 대한 투자를 확대하고 있습니다.
        - 앞으로의 청년 취업 기회는 **신재생에너지, 친환경 교통, 기후기술 개발 분야**에서 크게 늘어날 것으로 전망됩니다.
        """)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.header("결론")
        st.write("""
        기후위기는 위기이자 기회입니다.
        
        산업 구조 전환 속에서 청년들이 **친환경·녹색 분야 전공 및 직업 역량**을 갖춘다면 안정적이고 유망한 일자리를 확보할 수 있습니다.
        
        따라서 기후를 고려한 진로 선택은 단순한 환경 보호 차원이 아니라 **미래 취업 전략**이기도 합니다.
        """)
    
    st.markdown("---")
    st.caption("출처: NASA GISTEMP, 고용노동부 보도자료, 한국직업능력연구원(YPEC), 기후변화행동연구소, 기후환경포털 등")

def display_future_jobs_page():
    """Render the content for the promising future jobs page."""
    st.title("💼 미래 유망 녹색 직업")
    st.markdown("기후 변화 대응은 새로운 직업 시장을 열고 있습니다. 아래는 주목받는 유망 직업 분야입니다.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.header("☀️ 재생에너지 분야")
            st.image("https://images.unsplash.com/photo-1558495033-69b14739265f?q=80&w=2070&auto=format&fit=crop", caption="풍력 발전소의 해질녘")
            with st.expander("**1. 태양광/풍력 발전 설비 기술자**"):
                st.markdown("""
                - **주요 업무**: 태양광 패널, 풍력 터빈 등 발전 설비의 설치, 유지보수, 성능 모니터링
                - **필요 역량**: 전기/전자 공학 지식, 기계 설비 이해, 현장 실무 능력, 드론 조종 능력(점검용)
                - **예상 연봉**: 초봉 4,000 ~ 6,000만원, 경력에 따라 1억 이상 가능
                """)
            with st.expander("**2. 에너지 저장 시스템(ESS) 전문가**"):
                st.markdown("""
                - **주요 업무**: 생산된 전력을 저장하고 필요할 때 공급하는 ESS의 설계, 구축, 운영
                - **필요 역량**: 화학공학(배터리), 전력 시스템, 데이터 분석, 소프트웨어 개발 능력
                - **예상 연봉**: 초봉 5,000 ~ 7,000만원, 기술 전문성에 따라 고연봉 형성
                """)
    
    with c2:
        with st.container(border=True):
            st.header("🚗 친환경 모빌리티 분야")
            st.image("https://images.unsplash.com/photo-1617886322207-6f504e7472c5?q=80&w=2070&auto=format&fit=crop", caption="충전 중인 전기차")
            with st.expander("**1. 전기차(EV) 배터리 엔지니어**"):
                st.markdown("""
                - **주요 업무**: 전기차의 핵심인 배터리 셀/모듈/팩의 연구개발, 설계, 테스트
                - **필요 역량**: 화학/재료공학, 전자회로, 열 관리 시스템에 대한 깊은 이해
                - **예상 연봉**: 초봉 6,000 ~ 8,000만원, R&D 분야 최고 수준 대우
                """)
            with st.expander("**2. 자율주행 소프트웨어 개발자**"):
                st.markdown("""
                - **주요 업무**: 차량의 센서 데이터(카메라, 라이다)를 처리하여 주행을 제어하는 알고리즘 개발
                - **필요 역량**: 컴퓨터 공학, 인공지능(AI), 머신러닝/딥러닝, C++/Python 프로그래밍
                - **예상 연봉**: 초봉 7,000만원 이상, IT 업계 최상위권
                """)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.header("🌳 환경 및 지속가능성 분야")
        st.image("https://images.unsplash.com/photo-1542601906-8e9a43a4925b?q=80&w=2070&auto=format&fit=crop", caption="숲에서 환경 데이터를 분석하는 모습")
        with st.expander("**1. ESG 컨설턴트**"):
            st.markdown("""
            - **주요 업무**: 기업이 환경(Environment), 사회(Social), 지배구조(Governance) 측면에서 지속가능한 경영을 하도록 전략 수립 및 자문
            - **필요 역량**: 경영/경제학, 환경 정책 이해, 데이터 분석, 커뮤니케이션 능력
            - **예상 연봉**: 컨설팅 펌 직급에 따라 상이 (초봉 5,000만원 이상)
            """)
        with st.expander("**2. 탄소배출권 거래 전문가**"):
            st.markdown("""
            - **주요 업무**: 기업이나 국가의 탄소 배출량을 분석하고, 배출권 거래 시장에서 매매를 통해 이익을 창출하거나 비용을 절감
            - **필요 역량**: 금융/경제, 환경 규제 지식, 시장 분석 능력, 리스크 관리
            - **예상 연봉**: 금융권 수준의 높은 연봉 (성과급 비중 높음)
            """)
        with st.expander("**3. 스마트팜 전문가**"):
            st.markdown("""
            - **주요 업무**: 정보통신기술(ICT)을 농업에 접목하여 생산 효율을 높이고, 기후 변화에 대응 가능한 농업 시스템을 구축
            - **필요 역량**: 농업생명과학, 데이터 분석, 사물인터넷(IoT) 기술, 자동제어 시스템 이해
            - **예상 연봉**: 초봉 4,000 ~ 5,500만원, 기술 창업 기회 다수
            """)

# ==============================================================================
# 3. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    """Main function to run the Streamlit app."""
    st.title("기후 변화와 취업 동향 통합 대시보드")

    # --- Data Loading ---
    if 'data_loaded' not in st.session_state:
        st.sidebar.title("데이터 로드 상태")
        with st.spinner("공식 공개 데이터를 불러오는 중입니다..."):
            climate_raw = fetch_gistemp_csv()
            st.session_state.climate_df = preprocess_dataframe(climate_raw if climate_raw is not None else get_sample_climate_data())
            co2_raw = fetch_noaa_co2_data()
            st.session_state.co2_df = preprocess_dataframe(co2_raw if co2_raw is not None else get_sample_co2_data())
            employment_raw = fetch_worldbank_employment()
            st.session_state.employment_df = preprocess_dataframe(employment_raw if employment_raw is not None else get_sample_employment_data())
            st.session_state.renewable_df = get_sample_renewable_data()
            st.session_state.data_loaded = True
            time.sleep(0.5)
            st.rerun()

    # --- Page Navigation ---
    page = st.sidebar.radio("📑 페이지 선택", ["🌏 공개 데이터 대시보드", "📄 가상 시나리오 분석", "💼 미래 유망 직업", "📝 분석 보고서"])

    if page == "🌏 공개 데이터 대시보드":
        display_public_data_tab(st.session_state.climate_df, st.session_state.co2_df, st.session_state.employment_df, st.session_state.renewable_df)
    elif page == "📄 가상 시나리오 분석":
        display_user_prompt_tab()
    elif page == "💼 미래 유망 직업":
        display_future_jobs_page()
    elif page == "📝 분석 보고서":
        display_report_page()

    with st.sidebar.expander("개발자 및 실행 환경 참고사항"):
        st.markdown("""
        - 이 앱은 NASA/NOAA/WorldBank 공개 API를 우선적으로 호출하며, 네트워크 실패 시 내장된 예시 데이터로 자동 전환됩니다.
        """)

if __name__ == "__main__":
    try:
        if os.path.exists(CONFIG["font_path"]):
            st.markdown(f"""
            <style>
            @font-face {{ font-family: 'PretendardCustom'; src: url('{CONFIG["font_path"]}') format('truetype'); }}
            html, body, [class*="css"] {{ font-family: 'PretendardCustom', Pretard, sans-serif; }}
            </style>""", unsafe_allow_html=True)
    except Exception: pass
    main()

