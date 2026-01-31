import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, silhouette_score, \
    mean_absolute_error
from sklearn.datasets import load_wine

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Auto EDA & ML", layout="wide", page_icon="📊")


# --- FUNKCJE POMOCNICZE ---
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Titanic":
        df = sns.load_dataset("titanic")
        df = df.dropna(subset=['age', 'embarked', 'fare'])
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        selected_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        df = df[selected_cols]

    elif dataset_name == "Iris":
        df = sns.load_dataset("iris")

    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target_class'] = data.target

    elif dataset_name == "Tips (Napiwki)":
        df = sns.load_dataset("tips")
        df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
        df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})
        df = df.select_dtypes(include=[np.number])

    return df


# --- INTERFEJS GŁÓWNY ---
st.title("Aplikacja zaliczeniowa EDA")
st.markdown("""
Aplikacja umożliwia eksploracyjną analizę danych oraz trenowanie podstawowych modeli uczenia maszynowego.
Autor: Maciej Wojciechowski
""")
st.divider()

# --- PANEL BOCZNY (SIDEBAR) ---
st.sidebar.header("Konfiguracja Danych")
dataset_name = st.sidebar.selectbox("Wybierz zbiór danych:", ["Titanic", "Iris", "Wine", "Tips"])

df = load_data(dataset_name)
st.sidebar.success(f"Załadowano zbiór: {dataset_name}")
st.sidebar.write(f"Liczba wierszy: {df.shape[0]}")
st.sidebar.write(f"Liczba kolumn: {df.shape[1]}")

# --- ZAKŁADKI GŁÓWNE ---
tab1, tab2 = st.tabs(["🔍 Eksploracyjna Analiza Danych (EDA)", "Uczenie Maszynowe (ML)"])

# ==========================================
# ZAKŁADKA 1: EDA
# ==========================================
with tab1:
    st.header(f"Analiza Eksploracyjna: {dataset_name}")

    # 1. Podgląd danych
    with st.expander("👀 Podgląd surowych danych (Dataframe)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # 2. Statystyki i Braki
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statystyki opisowe")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.subheader("Braki danych")
        nulls = df.isnull().sum()
        if nulls.sum() == 0:
            st.success("Brak pustych wartości w tym zbiorze!")
        else:
            st.warning("Wykryto braki danych:")
            st.dataframe(nulls[nulls > 0], use_container_width=True)

            # Heatmapa braków
            fig_null = px.imshow(df.isnull(), title="Mapa ciepła brakujących danych")
            st.plotly_chart(fig_null, use_container_width=True)

    st.divider()

    # 3. Wizualizacje Rozkładów
    st.subheader("Wizualizacja Zmiennych")

    viz_cols = st.columns([1, 3])
    with viz_cols[0]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Wybierz kolumnę do analizy:", numeric_cols)
        viz_type = st.radio("Typ wykresu:", ["Histogram", "Boxplot (Pudełkowy)"])

    with viz_cols[1]:
        if viz_type == "Histogram":
            fig = px.histogram(df, x=selected_col, nbins=30, title=f"Rozkład: {selected_col}",
                               color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(df, y=selected_col, title=f"Wykres pudełkowy: {selected_col}",
                         color_discrete_sequence=['#FF9900'])
            st.plotly_chart(fig, use_container_width=True)

    # 4. Korelacje
    st.divider()
    st.subheader("Macierz Korelacji")

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                             title="Mapa korelacji zmiennych numerycznych")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Za mało zmiennych numerycznych do policzenia korelacji.")

    # 5. Wykres Rozrzutu (Scatter)
    st.subheader("punkty Wykres rozrzutu (Scatter Plot)")
    sc1, sc2, sc3 = st.columns(3)
    x_axis = sc1.selectbox("Oś X", numeric_cols, index=0)
    y_axis = sc2.selectbox("Oś Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    color_var = sc3.selectbox("Kolor (opcjonalnie)", ["Brak"] + df.columns.tolist())

    color_arg = None if color_var == "Brak" else color_var
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color_arg, title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# ZAKŁADKA 2: UCZENIE MASZYNOWE (ML)
# ==========================================
with tab2:
    st.header("Moduł Uczenia Maszynowego")
    st.info("Wybierz algorytm, skonfiguruj zmienne i wytrenuj model.")

    # Wybór zadania
    task = st.selectbox("Wybierz typ zadania ML:", ["Klasyfikacja", "Regresja", "Klasteryzacja (K-Means)"])
    st.divider()

    # --- 1. KLASYFIKACJA ---
    if task == "Klasyfikacja":
        st.subheader("Konfiguracja Klasyfikacji (Regresja Logistyczna)")

        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Zmienna celu (y) - co przewidujemy?", df.columns)
        feature_cols = c2.multiselect("Zmienne objaśniające (X)", [c for c in numeric_cols if c != target_col])

        if st.button("Trenuj Klasyfikator"):
            if feature_cols and target_col:
                try:
                    X = df[feature_cols]
                    y = df[target_col]

                    # Czy y jest numeryczne czy tekstowe?
                    if y.dtype == 'object':
                        y = y.astype('category').cat.codes

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # WYNIKI
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Model wytrenowany pomyślnie! Dokładność: {acc:.2%}")

                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.write("**Macierz Pomyłek:**")
                        cm = confusion_matrix(y_test, preds)
                        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as e:
                    st.error(f"Błąd trenowania: {e}")
            else:
                st.warning("Wybierz zmienną celu i przynajmniej jedną cechę (X).")

    # --- 2. REGRESJA ---
    elif task == "Regresja":
        st.subheader("Konfiguracja Regresji Liniowej")

        c1, c2 = st.columns(2)
        # Filtrujemy tylko numeryczne, bo regresja liniowa wymaga liczb
        target_col = c1.selectbox("Zmienna celu (y) - wartość ciągła", numeric_cols)
        feature_cols = c2.multiselect("Zmienne objaśniające (X)", [c for c in numeric_cols if c != target_col])

        if st.button("Trenuj Regresor"):
            if feature_cols and target_col:
                try:
                    X = df[feature_cols]
                    y = df[target_col]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # METRYKI
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("R² Score", f"{r2:.4f}")
                    m2.metric("MAE", f"{mae:.4f}")
                    m3.metric("MSE", f"{mse:.4f}")

                    # WYKRES RZECZYWISTE vs PRZEWIDYWANE
                    results_df = pd.DataFrame({'Rzeczywiste': y_test, 'Przewidywane': preds})
                    fig_reg = px.scatter(results_df, x='Rzeczywiste', y='Przewidywane',
                                         title="Wartości Rzeczywiste vs Przewidywane", trendline="ols")
                    st.plotly_chart(fig_reg, use_container_width=True)

                except Exception as e:
                    st.error(f"Błąd: {e}. Upewnij się, że dane są numeryczne.")
            else:
                st.warning("Wybierz zmienne X i y.")

    # --- 3. KLASTERYZACJA ---
    elif task == "Klasteryzacja (K-Means)":
        st.subheader("Algorytm K-Means")

        clus_cols = st.multiselect("Wybierz zmienne do klasteryzacji", numeric_cols, default=numeric_cols[:2])
        k_clusters = st.slider("Liczba klastrów (k)", 2, 8, 3)

        if st.button("Uruchom K-Means"):
            if len(clus_cols) >= 2:
                X = df[clus_cols]

                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)

                # Dodajemy klastry do kopii dataframe dla wizualizacji
                X_vis = X.copy()
                X_vis['Cluster'] = clusters.astype(str)

                # Metryka Silhouette
                sil_score = silhouette_score(X, clusters)
                st.metric("Silhouette Score (Jakość podziału)", f"{sil_score:.3f}")
                if sil_score > 0.5:
                    st.success("Dobre dopasowanie klastrów!")

                # Wizualizacja 2D
                st.write("Wizualizacja klastrów (na podstawie 2 pierwszych wybranych cech):")
                fig_clus = px.scatter(X_vis, x=clus_cols[0], y=clus_cols[1], color='Cluster',
                                      title=f"Klastry (k={k_clusters})", symbol='Cluster')
                st.plotly_chart(fig_clus, use_container_width=True)
            else:
                st.warning("Wybierz przynajmniej 2 zmienne numeryczne do klasteryzacji.")