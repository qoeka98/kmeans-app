import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import os
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📌 사용자 폰트 등록 함수
@st.cache_data
def register_custom_font():
    font_path = os.path.join(os.getcwd(), 'custom_fonts', 'NanumSquareRound.ttf')  # 폰트 경로 지정
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        rc('font', family='NanumSquareRound')  # 한글 폰트 적용
    else:
        st.warning("⚠️ 사용자 지정 폰트를 찾을 수 없습니다. 기본 폰트가 사용됩니다.")

# 📌 앱 메인 실행 함수
def main():
    register_custom_font()

    st.title('K-Means Clustering App')

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

        #  NaN 값 처리
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

        # WCSS 그래프 출력
        fig1, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='b')
        ax.set_xlabel('클러스터 개수 (k)')
        ax.set_ylabel('WCSS 값')
        ax.set_title('엘보우 메서드')

        st.pyplot(fig1)

        k = st.number_input('원하는 클러스터 수 입력', min_value=2, max_value=max_k)

        df_new.dropna(inplace=True)

        # KMeans 모델 생성 및 클러스터링 수행
        kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_new)

        df = df.loc[df_new.index]
        df['Group'] = cluster_labels

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)

if __name__ == '__main__':
    main()
