import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # NaN 처리용
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import platform

import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)


if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    #한글이면 폰트 설치 작업을 해줘야한다



def main():
    fontRegistered()
    plt.rc('font',family='MaruBuri-Bold.otf')
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

        df_new = pd.DataFrame(index=df.index)  #  df와 같은 인덱스 유지

        # 4. 컬럼 타입 확인 및 변환
        for column in selected_columns:
            if is_integer_dtype(df[column]) or is_float_dtype(df[column]):
                df_new[column] = df[column]  # 숫자형 데이터는 그대로 추가
            elif is_object_dtype(df[column]):
                if df[column].nunique() <= 2:
                    # 레이블 인코딩 (이진 데이터)
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else:
                    #  OneHotEncoder 적용
                    encoder = OneHotEncoder(sparse_output=False)
                    transformed_data = encoder.fit_transform(df[[column]])

                    # `get_feature_names_out()`을 사용하여 컬럼명 가져오기
                    column_names = encoder.get_feature_names_out([column])

                    # 변환된 데이터를 데이터프레임으로 변환 후 추가
                    df_transformed = pd.DataFrame(transformed_data, columns=column_names, index=df.index)
                    df_new = pd.concat([df_new, df_transformed], axis=1)

            else:
                st.text(f'⚠️ {column} 컬럼은 K-Means에 사용할 수 없어 제외됩니다.')

        #  선택한 컬럼이 없는 경우 에러 메시지 출력 후 종료
        if df_new.empty:
            st.error("⚠️ 데이터프레임이 비어 있습니다. 컬럼을 선택하세요.")
            return

        #  모든 데이터를 숫자형으로 변환
        df_new = df_new.apply(pd.to_numeric, errors='coerce')

        st.info('K-Means를 수행하기 위한 변환된 데이터 프레임입니다.')
        st.dataframe(df_new)

        #  NaN 값 처리: 평균값으로 대체
        imputer = SimpleImputer(strategy='mean')  
        df_new[:] = imputer.fit_transform(df_new)

        st.info('NaN 값이 처리된 데이터 프레임')
        st.dataframe(df_new.isna().sum())  # NaN이 완전히 사라졌는지 확인

        # 5. 최적의 k값을 찾기 위한 WCSS 계산
        st.subheader('최적의 K값을 찾기 위해 WCSS를 계산합니다.')
        
        # 데이터 개수에 따라 최대 k값 설정
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
        ax.set_xlabel('클러스터 갯수')
        ax.set_ylabel('WCSS값')
        ax.set_title('WCSS vs. Number of Clusters')

        st.pyplot(fig1)

        st.text('원하는 클러스터링(그룹) 갯수를 입력하세요')

        k = st.number_input('숫자 입력', min_value=2, max_value=max_k)

        #  NaN 값 제거 (필수)
        df_new.dropna(inplace=True)

        #  df_new와 df의 인덱스 동기화 확인
        st.text(f"df_new 데이터 개수: {df_new.shape[0]}, df 데이터 개수: {df.shape[0]}")
        
        #  KMeans 모델 생성 및 클러스터링 수행
        kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_new)

        #  df_new와 같은 인덱스를 가진 df에 Group 컬럼 추가
        df = df.loc[df_new.index]  # df와 df_new의 인덱스를 동기화
        df['Group'] = cluster_labels

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)

if __name__ == '__main__':
    main()
