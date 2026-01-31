import ssl
# --- FIX SSL DLA MACOS/CLOUD ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, silhouette_score, mean_absolute_error
from sklearn.datasets import load_wine

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Auto EDA & ML", layout="wide", page_icon="📊")

# --- FUNKCJE POMOCNICZE ---
@st.cache_data
def load_data(dataset_name):
    # 1. Inicjalizujemy pusty DataFrame na wypadek błędu
    df = pd.DataFrame()
    
    try:
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

        # POPRAWKA: Ujednolicona nazwa "Tips"
        elif dataset_name == "Tips":
            df = sns.load_dataset("tips")
            # Bezpieczne mapowanie (zamiana na string przed mapowaniem)
            df['sex'] = df['sex'].astype(str).map({'Male': 0, 'Female': 1})
            df['smoker'] = df['smoker'].astype(str).map({'Yes': 1, 'No': 0})
            # Wybieramy tylko numeryczne, żeby uniknąć błędów
            df = df.select_dtypes(include=[np.number])

    except Exception as e:
        st.error(f"Wystąpił błąd podczas ładowania danych: {e}")
        return pd.DataFrame() # Zwróć pusty w razie błędu
        
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
# POPRAWKA: Nazwa "Tips" musi być identyczna jak w funkcji load_data
dataset_name = st.sidebar.selectbox("Wybierz zbiór danych:", ["Titanic", "Iris", "Wine", "Tips"])

df = load_data(dataset_name)

# Sprawdzenie czy dane się załadowały
if df.empty:
    st.warning("Nie załadowano danych lub wystąpił błąd. Spróbuj wybrać inny zbiór.")
    st.stop() # Zatrzymuje dalsze wykonywanie kodu, żeby nie sypało błędami

st.sidebar.success(f"Załadowano zbiór: {dataset_name}")
st.sidebar.write(f"Liczba wierszy: {df.shape[0]}")
st.sidebar.write(f"Liczba kolumn: {df.shape[1]}")

# --- GLOBALNE ZMIENNE ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# --- ZAKŁADKI GŁÓWNE ---
tab1, tab2 = st.tabs(["🔍 Eksploracyjna Analiza Danych (EDA)", "🤖 Uczenie Maszynowe (ML)"])

# ==========================================
# ZAKŁADKA 1: EDA
# ==========================================
with tab1:
    st.header(f"Analiza Eksploracyjna: {dataset_name}")

    with st.expander("👀 Podgląd surowych danych (Dataframe)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

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
            if not df.isnull().all().all():
                fig_null = px.imshow(df.isnull(), title="Mapa ciepła brakujących danych")
                st.plotly_chart(fig_null, use_container_width=True)

    st.divider()

    st.subheader("Wizualizacja Zmiennych")
    viz_cols = st.columns([1, 3])
    with viz_cols[0]:
        if numeric_cols:
            selected_col = st.selectbox("Wybierz kolumnę do analizy:", numeric_cols)
            viz_type = st.radio("Typ wykresu:", ["Histogram", "Boxplot (Pudełkowy)"])
        else:
            selected_col = None
            st.info("Brak kolumn numerycznych.")

    with viz_cols[1]:
        if selected_col:
            if viz_type == "Histogram":
                fig = px.histogram(df, x=selected_col, nbins=30, title=f"Rozkład: {selected_col}",
                                   color_discrete_sequence=['#3366CC'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.box(df, y=selected_col, title=f"Wykres pudełkowy: {selected_col}",
                             color_discrete_sequence=['#FF9900'])
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Macierz Korelacji")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                             title="Mapa korelacji zmiennych numerycznych")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Za mało zmiennych numerycznych do policzenia korelacji.")

    st.subheader("Wykres rozrzutu (Scatter Plot)")
    if len(numeric_cols) >= 2:
        sc1, sc2, sc3 = st.columns(3)
        x_axis = sc1.selectbox("Oś X", numeric_cols, index=0)
        idx_y = 1 if len(numeric_cols) > 1 else 0
        y_axis = sc2.selectbox("Oś Y", numeric_cols, index=idx_y)
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

    task = st.selectbox("Wybierz typ zadania ML:", ["Klasyfikacja", "Regresja", "Klasteryzacja (K-Means)"])
    st.divider()

    # --- 1. KLASYFIKACJA ---
    if task == "Klasyfikacja":
        st.subheader("Konfiguracja Klasyfikacji (Regresja Logistyczna)")
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Zmienna celu (y) - co przewidujemy?", df.columns)
        available_features = [c for c in numeric_cols if c != target_col]
        feature_cols = c2.multiselect("Zmienne objaśniające (X)", available_features)

        if st.button("Trenuj Klasyfikator"):
            if feature_cols and target_col:
                try:
                    X = df[feature_cols]
                    y = df[target_col]
                    if y.dtype == 'object' or hasattr(y, 'cat'):
                        y = y.astype('category').cat.codes

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
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
                st.warning("Wybierz zmienną celu i cechy (X).")

    # --- 2. REGRESJA ---
    elif task == "Regresja":
        st.subheader("Konfiguracja Regresji Liniowej")
        c1, c2 = st.columns(2)
        if len(numeric_cols) < 2:
            st.error("Za mało zmiennych numerycznych do regresji.")
        else:
            target_col = c1.selectbox("Zmienna celu (y)", numeric_cols)
            available_features = [c for c in numeric_cols if c != target_col]
            feature_cols = c2.multiselect("Zmienne objaśniające (X)", available_features)

            if st.button("Trenuj Regresor"):
                if feature_cols and target_col:
                    try:
                        X = df[feature_cols]
                        y = df[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        mse = mean_squared_error(y_test, preds)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("R² Score", f"{r2:.4f}")
                        m2.metric("MAE", f"{mae:.4f}")
                        m3.metric("MSE", f"{mse:.4f}")

                        results_df = pd.DataFrame({'Rzeczywiste': y_test, 'Przewidywane': preds})
                        try:
                            # Próba rysowania z linią trendu (wymaga statsmodels)
                            fig_reg = px.scatter(results_df, x='Rzeczywiste', y='Przewidywane',
                                                 title="Wartości Rzeczywiste vs Przewidywane", trendline="ols")
                        except:
                            # Fallback bez linii trendu
                            fig_reg = px.scatter(results_df, x='Rzeczywiste', y='Przewidywane',
                                                 title="Wartości Rzeczywiste vs Przewidywane")
                        st.plotly_chart(fig_reg, use_container_width=True)
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                else:
                    st.warning("Wybierz zmienne X i y.")

    # --- 3. KLASTERYZACJA ---
    elif task == "Klasteryzacja (K-Means)":
        st.subheader("Algorytm K-Means")
        default_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        clus_cols = st.multiselect("Wybierz zmienne do klasteryzacji", numeric_cols, default=default_cols)
        k_clusters = st.slider("Liczba klastrów (k)", 2, 8, 3)

        if st.button("Uruchom K-Means"):
            if len(clus_cols) >= 2:
                try:
                    X = df[clus_cols]
                    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X)
                    X_vis = X.copy()
                    X_vis['Cluster'] = clusters.astype(str)
                    sil_score = silhouette_score(X, clusters)
                    st.metric("Silhouette Score", f"{sil_score:.3f}")

                    st.write("Wizualizacja klastrów:")
                    fig_clus = px.scatter(X_vis, x=clus_cols[0], y=clus_cols[1], color='Cluster',
                                          title=f"Klastry (k={k_clusters})", symbol='Cluster')
                    st.plotly_chart(fig_clus, use_container_width=True)
                except Exception as e:
                    st.error(f"Błąd klasteryzacji: {e}")
            else:
                st.warning("Wybierz przynajmniej 2 zmienne.")
