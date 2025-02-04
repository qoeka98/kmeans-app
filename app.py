import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
import platform
import matplotlib.font_manager as fm
from matplotlib import rc

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📌 OS별 기본 한글 폰트 지정
def get_default_font():
    system_name = platform.system()
    if system_name == "Windows":
        return "Malgun Gothic"
    elif system_name == "Darwin":  # macOS
        return "AppleGothic"
    else:  # Linux (Ubuntu 등)
        return "NanumGothic"

# 📌 사용자 폰트 등록 함수
@st.cache_data
def register_font():
    font_path = os.path.join(os.getcwd(), 'custom_fonts', 'NanumSquareRoundR.ttf')  # 폰트 경로
    default_font = get_default_font()

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)  # Matplotlib에 폰트 등록
        fm._load_fontmanager(try_read_cache=False)  # 캐시 무시하고 강제 로드
        rc('font', family='NanumSquareRoundR')  # Matplotlib에서 한글 폰트 적용
        print("✅ NanumSquareRoundR 폰트가 정상적으로 등록되었습니다!")
    else:
        st.warning(f"⚠️ NanumSquareRoundR 폰트를 찾을 수 없습니다. 기본 폰트({default_font})를 사용합니다.")
        rc('font', family=default_font)  # OS별 기본 폰트 적용

def main():
    register_font()  # 한글 폰트 적용
    st.title('K-Means 클러스터링 앱')

    # 1. CSV 파일 업로드
    file = st.file_uploader('CSV 파일 업로드', type=['csv'])

    if file is not None:
        # 2. 데이터 불러오기
        df = pd.read_csv(file)
        st.dataframe(df.head())

        st.info('NaN 값이 있는 행을 삭제합니다.')
        df.dropna(inplace=True)

        # 3. 유저가 컬럼을 선택할 수 있도록 함
        st.info('K-Means 클러스터링에 사용할 컬럼을 선택해주세요.')
        selected_columns = st.multiselect('컬럼 선택', df.columns)

        df_new = pd.DataFrame(index=df.index)  # df와 같은 인덱스 유지

        # 4. 컬럼 타입 확인 및 변환
        for column in selected_columns:
            if is_integer_dtype(df[column]) or is_float_dtype(df[column]):
                df_new[column] = df[column]  # 숫자형 데이터는 그대로 추가
            elif is_object_dtype(df[column]):
                if df[column].nunique() <= 2:
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else:
                    encoder = OneHotEncoder(sparse_output=False)
                    transformed_data = encoder.fit_transform(df[[column]])

                    column_names = encoder.get_feature_names_out([column])
                    df_transformed = pd.DataFrame(transformed_data, columns=column_names, index=df.index)
                    df_new = pd.concat([df_new, df_transformed], axis=1)

            else:
                st.text(f'⚠️ {column} 컬럼은 K-Means에 사용할 수 없어 제외됩니다.')

        if df_new.empty:
            st.error("⚠️ 데이터프레임이 비어 있습니다. 컬럼을 선택하세요.")
            return

        df_new = df_new.apply(pd.to_numeric, errors='coerce')

        st.info('K-Means를 수행하기 위한 변환된 데이터 프레임입니다.')
        st.dataframe(df_new)

        # NaN 값 처리
        imputer = SimpleImputer(strategy='mean')
        df_new[:] = imputer.fit_transform(df_new)

        st.info('NaN 값이 처리된 데이터 프레임')
        st.dataframe(df_new.isna().sum())

        # 5. 최적의 k값을 찾기 위한 WCSS 계산
        st.subheader('최적의 K값을 찾기 위해 WCSS를 계산합니다.')

        st.text(f'데이터 개수: {df_new.shape[0]}개')
        max_k = st.slider('K값 선택 (최대 그룹 개수)', min_value=2, max_value=min(10, df_new.shape[0]))

        # WCSS 계산
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        # 📌 WCSS 그래프 출력 (한글 깨짐 방지)
        fig, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='b')

        # ✅ 한글 깨짐 방지를 위한 폰트 직접 지정
        default_font = get_default_font()
        ax.set_xlabel('클러스터 개수 (k)', fontsize=12, fontweight='bold', fontname=default_font)
        ax.set_ylabel('WCSS 값', fontsize=12, fontweight='bold', fontname=default_font)
        ax.set_title('엘보우 메서드', fontsize=14, fontweight='bold', fontname=default_font)

        st.pyplot(fig)

if __name__ == '__main__':
    main()
