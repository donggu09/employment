# streamlit_app.py
"""
Streamlit 대시보드 (한국어)
- 공개 데이터 대시보드: 기후(온도 이상치) + 고용(산업별 고용비율(World Bank)) 연계 시각화
  - 출처 주석:
    * NASA GISTEMP (Global temperature anomalies CSV): https://data.giss.nasa.gov/gistemp/
      (직접 CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)
    * World Bank - Employment in industry (% of total employment): https://data.worldbank.org/indicator/SL.IND.EMPL.ZS
    * ILO / Green jobs 참고자료: https://ilostat.ilo.org/data/ , https://www.ilo.org/
- 사용자 입력(프롬프트 텍스트 기반) 대시보드: 제공된 리포트 텍스트에서 생성한 예시 데이터로 시각화
- 구현 규칙 준수:
  - 데이터 표준화: date, value, group(optional)
  - 전처리: 결측/형변환/중복/미래데이터 제거
  - 캐싱: @st.cache_data 사용
  - CSV 다운로드 버튼 제공
  - 폰트: /fonts/Pretendard-Bold.ttf 사용 시 적용 시도
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import datetime
from dateutil import parser
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

TODAY = datetime.date(2025, 9, 16)  # 명시된 로컬 날짜 (developer message)
# NOTE: 앱은 실행 환경의 실제 날짜를 query하지 않고 위 TODAY를 기준으로 "오늘 이후" 데이터 제거 규칙을 적용합니다.

st.set_page_config(page_title="기후와 취업 대시보드", layout="wide")

# --- 폰트 적용 시도 ---
def inject_pretendard():
    css = ""
    try:
        # relative path /fonts/Pretendard-Bold.ttf (if Codespaces repo includes it)
        css = f"""
        <style>
        @font-face {{
            font-family: 'PretendardCustom';
            src: url('/fonts/Pretendard-Bold.ttf') format('truetype');
            font-weight: 700;
            font-style: normal;
        }}
        html, body, [class*="css"]  {{
            font-family: PretendardCustom, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # 실패 시 무시 (자동 생략)
        pass

inject_pretendard()

# --- 유틸리티: 미래(오늘 이후) 날짜 제거 ---
def remove_future_dates(df, date_col="date"):
    if date_col not in df.columns:
        return df
    def to_date_safe(x):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return None
    df['_parsed_date'] = df[date_col].apply(to_date_safe)
    df = df[df['_parsed_date'].notnull()]
    df = df[df['_parsed_date'] <= TODAY]
    df = df.drop(columns=['_parsed_date'])
    return df

# --- 캐시된 다운로드 helpers ---
@st.cache_data(ttl=3600)
def fetch_gistemp_csv():
    """
    NASA GISTEMP CSV (global monthly anomalies)
    공식 출처: https://data.giss.nasa.gov/gistemp/
    안정적인 표형식 CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        content = r.content.decode('utf-8', errors='replace')
        # GISTEMP table has header lines; pandas can read skipping comment lines starting with 'Year'
        df = pd.read_csv(io.StringIO(content), skiprows=1)
        # transform: months in columns -> long format with date
        df = df.rename(columns={c: c.strip() for c in df.columns})
        df_long = df.melt(id_vars=['Year'], var_name='Month', value_name='Anomaly')
        # Month might be 'Jan', 'Feb', etc. Build date
        month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,'J-D':None,'DJF':None}
        def to_date(row):
            m = row['Month']
            y = int(row['Year'])
            if isinstance(m, str):
                m_clean = m.strip()
                mon = month_map.get(m_clean, None)
                if mon:
                    return datetime.date(y, mon, 1)
            return None
        df_long['date'] = df_long.apply(to_date, axis=1)
        df_long = df_long[df_long['date'].notnull()].copy()
        df_long = df_long[['date','Anomaly']].rename(columns={'Anomaly':'value'})
        # remove future dates
        df_long = remove_future_dates(df_long, 'date')
        df_long['group'] = '지구 평균 기온 이상치(℃)'
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
        df_long = df_long.dropna(subset=['value'])
        return df_long.sort_values('date')
    except Exception as e:
        # 실패 -> None to indicate fallback
        return None

@st.cache_data(ttl=3600)
def fetch_worldbank_employment():
    """
    World Bank API: Employment in industry (% of total employment) - indicator SL.IND.EMPL.ZS
    API: http://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS?format=json&per_page=20000
    출처: https://data.worldbank.org/indicator/SL.IND.EMPL.ZS
    """
    api = "http://api.worldbank.org/v2/country/all/indicator/SL.IND.EMPL.ZS"
    params = {'format':'json','per_page':20000}
    try:
        r = requests.get(api, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        # data[1] has records
        records = data[1]
        rows = []
        for rec in records:
            country = rec.get('country', {}).get('value')
            year = rec.get('date')
            val = rec.get('value')
            if val is None:
                continue
            try:
                d = datetime.date(int(year),1,1)
            except Exception:
                continue
            rows.append({'date':d, 'country':country, 'value':float(val)})
        df = pd.DataFrame(rows)
        # remove future dates
        df = df[df['date'] <= TODAY]
        # Standardize: date,value,group(country)
        df = df.rename(columns={'country':'group'})
        return df
    except Exception:
        return None

# --- 예시(대체) 데이터 생성 함수 ---
def sample_climate_data():
    # 간단한 예시: 연별 평균 이상치 (2010-2024)
    years = list(range(2010, 2025))
    dates = [datetime.date(y,1,1) for y in years]
    # synthetic increasing anomalies
    values = np.round(np.linspace(0.6, 1.2, len(years)) + np.random.normal(0,0.05,len(years)), 3)
    df = pd.DataFrame({'date':dates, 'value':values})
    df['group'] = '지구 평균 기온 이상치(℃)'
    return df

def sample_employment_data():
    # 예시: 산업별 고용비율(연도별) - 간단화: industry% for Korea and OECD-average
    years = list(range(2015, 2025))
    rows = []
    for y in years:
        rows.append({'date':datetime.date(y,1,1), 'group':'한국 산업 고용 비율(%)', 'value': float(25.0 - (2024-y)*0.2 + np.random.normal(0,0.5))})
        rows.append({'date':datetime.date(y,1,1), 'group':'세계 산업 고용 비율(%)', 'value': float(22.0 - (2024-y)*0.1 + np.random.normal(0,0.5))})
    return pd.DataFrame(rows)

# --- 공개 데이터 로드 (시도 -> 재시도 -> 실패 시 예시 데이터 할당) ---
st.sidebar.title("데이터 로드 상태")
with st.spinner("공식 공개 데이터 불러오는 중..."):
    gistemp_df = fetch_gistemp_csv()
    wb_df = fetch_worldbank_employment()

# retry logic: if None, retry once
if gistemp_df is None:
    gistemp_df = fetch_gistemp_csv()
if wb_df is None:
    wb_df = fetch_worldbank_employment()

# If still None -> fallback example and show notice
if gistemp_df is None:
    st.sidebar.error("NASA GISTEMP 데이터 로드 실패: 예시(대체) 데이터 사용")
    gistemp_df = sample_climate_data()
else:
    st.sidebar.success("NASA GISTEMP 데이터 로드 성공")

if wb_df is None or wb_df.empty:
    st.sidebar.error("World Bank 고용(산업별) 데이터 로드 실패: 예시(대체) 데이터 사용")
    wb_df = sample_employment_data()
else:
    st.sidebar.success("World Bank 데이터 로드 성공 (산업별 고용)")

# --- 공개 데이터 전처리 공통 함수 ---
@st.cache_data
def preprocess_public_climate(df):
    dfc = df.copy()
    # ensure date dtype
    dfc['date'] = pd.to_datetime(dfc['date'])
    dfc = dfc.sort_values('date')
    # remove duplicates
    dfc = dfc.drop_duplicates(subset=['date'])
    # remove future dates (already applied but safe)
    dfc = dfc[dfc['date'].dt.date <= TODAY]
    # fill missing values by interpolation
    if dfc['value'].isnull().any():
        dfc['value'] = dfc['value'].interpolate().fillna(method='bfill').fillna(method='ffill')
    return dfc

@st.cache_data
def preprocess_public_employment(df):
    dfe = df.copy()
    dfe['date'] = pd.to_datetime(dfe['date'])
    dfe = dfe.sort_values(['group','date']) if 'group' in dfe.columns else dfe.sort_values(['date'])
    # dedupe
    dfe = dfe.drop_duplicates(subset=['date','group'] if 'group' in dfe.columns else ['date'])
    # remove future dates
    dfe = dfe[dfe['date'].dt.date <= TODAY]
    dfe['value'] = pd.to_numeric(dfe['value'], errors='coerce')
    # fill missing per group
    if 'group' in dfe.columns:
        dfe['value'] = dfe.groupby('group')['value'].apply(lambda s: s.interpolate().fillna(method='bfill').fillna(method='ffill'))
    else:
        dfe['value'] = dfe['value'].interpolate().fillna(method='bfill').fillna(method='ffill')
    return dfe

climate_df = preprocess_public_climate(gistemp_df)
employment_df = preprocess_public_employment(wb_df)

# --- 탭: 공개 데이터 대시보드 ---
tab1, tab2 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력(리포트 기반) 대시보드"])

with tab1:
    st.header("공식 공개 데이터 대시보드")
    st.markdown("**데이터 출처**: NASA GISTEMP (기후 이상치), World Bank (산업별 고용 비율).")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("지구 평균 온도 이상치 (시간 흐름)")
        # line chart via plotly
        fig = px.line(climate_df, x='date', y='value', title='지구 평균 온도 이상치 (월별)', labels={'date':'연도','value':'이상치 (℃)'}, hover_data={'date':True,'value':True})
        fig.update_layout(legend_title_text=None)
        st.plotly_chart(fig, use_container_width=True)
        # CSV download
        csv = climate_df.to_csv(index=False).encode('utf-8')
        st.download_button("기후 전처리 데이터 다운로드 (CSV)", csv, file_name="climate_preprocessed.csv", mime="text/csv")
    with col2:
        st.subheader("산업별 고용 비율 개요")
        # summarize latest year per country/group
        # employment_df expected to have group=country; show top countries latest year
        try:
            latest = employment_df[employment_df['date']==employment_df['date'].max()]
            top = latest.sort_values('value', ascending=False).head(10)
            st.table(top[['group','value']].rename(columns={'group':'국가/그룹','value':'산업 고용 비율(%)'}).reset_index(drop=True))
        except Exception:
            st.write("고용 데이터가 충분하지 않아 요약을 표시할 수 없습니다.")
        csv2 = employment_df.to_csv(index=False).encode('utf-8')
        st.download_button("고용 전처리 데이터 다운로드 (CSV)", csv2, file_name="employment_preprocessed.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("기후(온도 이상치) vs 산업 고용(연도별 비교)")
    # For comparison, aggregate employment by year global median or specific groups
    # We'll create a synthetic aggregated employment timeseries if original wb df is country-level
    try:
        emp_agg = employment_df.groupby('date')['value'].median().reset_index().rename(columns={'value':'industry_employment_median'})
        merged = pd.merge(climate_df.groupby(climate_df['date'].dt.to_period('Y')).mean().reset_index(), emp_agg, left_on='date', right_on='date', how='inner')
        # fallback if merge empty
        if merged.empty:
            # aggregate climate annually
            c_ann = climate_df.copy()
            c_ann['year'] = c_ann['date'].dt.year
            c_ann_agg = c_ann.groupby('year')['value'].mean().reset_index()
            e_ann = employment_df.copy()
            e_ann['year'] = e_ann['date'].dt.year
            e_ann_agg = e_ann.groupby('year')['value'].median().reset_index()
            merged = pd.merge(c_ann_agg, e_ann_agg, on='year', how='inner')
            merged = merged.rename(columns={'value_x':'temp_anomaly','value_y':'industry_employment_median'})
            fig2 = px.line(merged, x='year', y=['temp_anomaly','industry_employment_median'], labels={'value':'값', 'variable':'지표'}, title='연도별: 기후 이상치 vs 산업 고용(중앙값)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.line(merged, x='date', y=['value','industry_employment_median'], labels={'value':'기후 이상치(평균)','industry_employment_median':'산업 고용(중앙값)'}, title='연도별 비교 (연 단위)')
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.write("비교 그래프 생성에 실패했습니다. (데이터 포맷 차이)")

    st.markdown("**API 실패 시 안내**: 공개 API 호출이 실패하면 예시(대체) 데이터를 자동 사용하며, 사이드바에 안내 메시지가 표시됩니다.")

with tab2:
    st.header("사용자 입력 대시보드 (제공된 리포트 텍스트 기반)")
    st.markdown("입력: 보고서 제목 및 본문(프롬프트로 제공된 텍스트)을 기반으로 자동 생성한 데이터만 사용합니다.")
    # Build synthetic datasets derived from the provided report content (사용자 입력 데이터만 사용)
    # 1) 녹색 일자리 vs 화석연료 일자리 변화 (2018-2024)
    years = list(range(2018, 2025))
    green_change = [5,8,12,15,18,22,26]  # 누적(예시) 녹색 일자리 증가(단위: 천명 또는 상대지수)
    fossil_change = [0,-2,-4,-6,-9,-12,-15]  # 감소
    df_jobs = pd.DataFrame({
        'date': [datetime.date(y,1,1) for y in years],
        '녹색_일자리_지표': green_change,
        '화석기반_일자리_지표': fossil_change
    })
    # standardize long format
    df_jobs_long = df_jobs.melt(id_vars=['date'], var_name='group', value_name='value')
    df_jobs_long = remove_future_dates(df_jobs_long, 'date')

    # 2) 전공별(친환경/에너지/IT/기타) 취업률 스냅샷 (예시)
    majors = ['친환경·에너지 관련 전공', 'IT 관련 전공', '전통 제조 전공', '기타 전공']
    employ_rates = [0.88, 0.82, 0.65, 0.70]  # 취업률 예시 (0-1)
    df_major = pd.DataFrame({'group':majors, 'value':[r*100 for r in employ_rates]})

    # Sidebar 자동 구성: 기간 필터(사용자 데이터에 맞춤), 스무딩(이동평균)
    st.sidebar.header("사용자 데이터 옵션")
    years_min = min(years)
    years_max = max(years)
    sel_start = st.sidebar.slider("기간 시작 연도", min_value=years_min, max_value=years_max, value=years_min)
    sel_end = st.sidebar.slider("기간 종료 연도", min_value=years_min, max_value=years_max, value=years_max)
    smoothing = st.sidebar.checkbox("시계열 스무딩(이동평균)", value=False)
    window = st.sidebar.slider("스무딩 윈도우(연)", 2, 5, 2) if smoothing else None

    # Filter df_jobs_long by selected years
    fj = df_jobs_long.copy()
    fj['year'] = fj['date'].apply(lambda d: d.year if isinstance(d, datetime.date) else pd.to_datetime(d).year)
    fj = fj[(fj['year'] >= sel_start) & (fj['year'] <= sel_end)]

    st.subheader("1) 녹색 전환에 따른 일자리 지표 (리포트 기반)")
    # plot line per group
    if smoothing and window:
        fj = fj.sort_values(['group','date'])
        fj['value_smooth'] = fj.groupby('group')['value'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        y_col = 'value_smooth'
        ylabel = '지표 (임의 단위)'
    else:
        y_col = 'value'
        ylabel = '지표 (임의 단위)'
    fig_jobs = px.line(fj, x='date', y=y_col, color='group', markers=True, labels={'date':'연도','value':'지표','value_smooth':'스무딩 지표','group':'구분'})
    fig_jobs.update_layout(title='녹색 일자리 증가 vs 화석 연료 기반 일자리 감소 (리포트 기반 예시 데이터)', yaxis_title=ylabel)
    st.plotly_chart(fig_jobs, use_container_width=True)
    st.markdown("설명: 이 그래프는 보고서 본문에 기술된 '녹색 일자리 증가'와 '화석연료 기반 일자리 감소'를 예시 수치로 만든 시계열입니다.")

    st.subheader("2) 전공별 취업률(스냅샷)")
    fig_pie = px.bar(df_major, x='group', y='value', labels={'group':'전공','value':'취업률(%)'}, text='value', title='전공별 취업률 (예시)')
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("설명: 보고서에서 언급된 '친환경·에너지 관련 전공자의 취업률이 평균보다 높다'는 문장을 수치화한 예시입니다.")

    # CSV downloads for user-data (preprocessed)
    st.download_button("사용자 입력(리포트 기반) - 전처리된 일자리 데이터 CSV", df_jobs_long.to_csv(index=False).encode('utf-8'), file_name="user_report_jobs.csv", mime="text/csv")
    st.download_button("사용자 입력(리포트 기반) - 전공별 취업률 CSV", df_major.to_csv(index=False).encode('utf-8'), file_name="user_report_majors.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("간단한 제언(리포트 기반 요약)")
    st.write("""
    1. 기후 정책·녹색 전환 가속화는 **녹색 일자리 증가**로 이어지므로 관련 전공·기술 습득이 권장됩니다.  
    2. 전통적 화석연료 산업은 구조조정과 축소가 예상되므로 대응 전략(재교육, 전환 교육)이 필요합니다.  
    3. IT 분야도 기후 데이터 분석·에너지 효율화 소프트웨어 등 새로운 기회를 제공하므로 융합 역량을 키우세요.
    """)

# --- 추가: 간단한 도움말 박스 (Kaggle API 안내 포함) ---
with st.expander("개발자용: Kaggle API 사용 안내 (필요 시)"):
    st.markdown("""
    - Kaggle에서 데이터 사용을 원할 경우 `kaggle` 패키지를 설치하고 API 토큰을 설정해야 합니다.
      1. Kaggle 계정 -> Account -> Create API token -> `kaggle.json` 다운로드  
      2. Codespaces나 로컬 환경: `~/.kaggle/kaggle.json` 경로에 파일을 저장하고 권한 `chmod 600 ~/.kaggle/kaggle.json` 부여  
      3. 예: `pip install kaggle` 후 `kaggle datasets download -d <dataset-owner>/<dataset-name>`  
    - 이 앱은 기본적으로 공식 기관(NASA, World Bank, ILO 등)의 공개 API/CSV를 우선 시도합니다.
    """)

# End of file
