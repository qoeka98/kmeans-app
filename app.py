import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# 글꼴 설치 os 시작 ---------------
import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)
# 글꼴 설치 os 끝 ---------------

# Streamlit 스타일 적용
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# 사이드바에 이미지 추가
st.sidebar.image("image.webp", use_container_width=True)  # 📌 사이드바 이미지 추가

# 메인 앱
def main():
    # 글꼴 추가 시작 ---------------
    fontRegistered()
    plt.rc('font', family='NanumBarunGothic')
    # 글꼴 추가 끝 ---------------
    st.title("🔍 K-Means Clustering App")
    st.markdown("##### 머신러닝 기반 K-Means 클러스터링을 간편하게 수행하세요.")

    # 1. CSV 파일 업로드
    st.sidebar.header("📂 CSV 파일 업로드")
    file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])

    if file is not None:
        # 2. 데이터 불러오기
        df = pd.read_csv(file)
        st.subheader("📜 데이터 미리보기")
        st.dataframe(df.head())

        # 2-1. NaN 처리
        st.sidebar.warning("❗ NaN 값을 포함한 행이 자동으로 삭제됩니다.")
        df.dropna(inplace=True)

        # 3. 컬럼 선택
        st.sidebar.subheader("🔧 K-Means 클러스터링 설정")
        selected_columns = st.sidebar.multiselect("📌 클러스터링에 사용할 컬럼을 선택하세요", df.columns)

        df_new = pd.DataFrame(index=df.index)

        # 4. 데이터 타입 변환
        for column in selected_columns:
            if is_integer_dtype(df[column]) or is_float_dtype(df[column]):
                df_new[column] = df[column]
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
                st.warning(f"⚠️ {column} 컬럼은 K-Means에 적합하지 않아 제외됩니다.")

        # 변환된 데이터 확인
        if df_new.empty:
            st.error("⚠️ 선택한 컬럼이 없거나 변환할 수 없는 데이터입니다.")
            return

        st.subheader("📊 변환된 데이터 미리보기")
        st.dataframe(df_new)

        # 5. NaN 처리 (평균값 대체)
        imputer = SimpleImputer(strategy="mean")
        df_new[:] = imputer.fit_transform(df_new)

        st.sidebar.success("✅ NaN 값이 평균값으로 대체되었습니다.")

        # 6. 최적의 K 값 찾기 (엘보우 메서드)
        st.subheader("🛠 최적의 K 값 찾기 (Elbow Method)")
        max_k = st.sidebar.slider("🔢 최대 K값 선택", min_value=2, max_value=min(10, df_new.shape[0]), value=5)

        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='b')
        ax.set_xlabel("클러스터 갯수")
        ax.set_ylabel("WCSS 값")
        ax.set_title("엘보우 메소드")

        st.pyplot(fig)

        # 7. 클러스터 개수 선택
        st.sidebar.subheader("🎯 K-Means 클러스터링 실행")
        k = st.sidebar.number_input("💡 클러스터 개수 선택 (K)", min_value=2, max_value=max_k, value=3)

        # 8. K-Means 실행
        if st.sidebar.button("🚀 클러스터링 실행"):
            df_new.dropna(inplace=True)
            kmeans = KMeans(n_clusters=k, random_state=4, n_init='auto')
            cluster_labels = kmeans.fit_predict(df_new)

            df = df.loc[df_new.index]  # 인덱스 동기화
            df["Cluster"] = cluster_labels

            st.success(f"✅ 클러스터링 완료! (K={k})")
            st.subheader("📌 클러스터링 결과 데이터")
            st.dataframe(df)

if __name__ == "__main__":
    main()
