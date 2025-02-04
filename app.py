import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # NaN 처리용
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import platform
import os
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    
    if system_name == 'Windows':
        plt.rcParams['font.family'] = ['Malgun Gothic']  # 윈도우 기본 한글 폰트
    elif system_name == 'Darwin':  # macOS
        plt.rcParams['font.family'] = ['AppleGothic']
    elif system_name == 'Linux':  # Linux 또는 Streamlit Cloud 환경
        plt.rcParams['font.family'] = ['NanumGothic']
    plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# ✅ 사용자 폰트 등록
def register_custom_font(font_path):
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"✅ 사용자 지정 폰트 적용됨: {font_prop.get_name()}")
    else:
        print("❌ 폰트 파일을 찾을 수 없습니다.")

# ✅ OS에 따라 기본 한글 폰트 설정
set_korean_font()

# ✅ 커스텀 폰트가 있으면 적용
register_custom_font("custom_fonts/MaruBuri-Bold.ttf")

def main():
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

        df_new = pd.DataFrame(index=df.index)  # ✅ df와 같은 인덱스 유지

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

        imputer = SimpleImputer(strategy='mean')  
        df_new[:] = imputer.fit_transform(df_new)

        st.subheader('최적의 K값을 찾기 위해 WCSS를 계산합니다.')
        max_k = st.slider('K값 선택 (최대 그룹 개수)', min_value=2, max_value=min(10, df_new.shape[0]))

        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        fig1, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='b')
        ax.set_xlabel('클러스터 갯수')
        ax.set_ylabel('WCSS값')
        ax.set_title('앨보우메서드')

        st.pyplot(fig1)

        st.text('원하는 클러스터링(그룹) 갯수를 입력하세요')

        k = st.number_input('숫자 입력', min_value=2, max_value=max_k)

        df_new.dropna(inplace=True)
        kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
        df['Group'] = kmeans.fit_predict(df_new)

        st.info('그룹 정보가 저장되었습니다.')
        st.dataframe(df)

if __name__ == '__main__':
    main()
