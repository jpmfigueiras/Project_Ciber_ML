# ============================================================
# models.py — Módulo de Treino e Avaliação de Modelos de ML
# ============================================================
# Este ficheiro contém funções reutilizáveis para:
# 1. Carregar e pré-processar o dataset
# 2. Treinar todos os modelos exigidos pelo enunciado
# 3. Avaliar e comparar os modelos
# 4. Guardar os modelos treinados
#
# Pode ser importado no notebook ou executado diretamente:
#   python src/models.py
# ============================================================

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE


# ============================================================
# CONSTANTES
# ============================================================

# Colunas contínuas que vão ser transformadas (Yeo-Johnson)
COLUNAS_CONTINUAS = [
    'e2c_total_bytes', 'e4_entropy_h', 'e5_entropy_c',
    'e6_time_char', 'e6b_flow_duration_ms',
    'e2_client_size', 'e2_client_record_len', 'e2_server_record_len'
]

# Colunas a remover antes do treino (não são features)
COLUNAS_META = ['label', 'split', 'taxonomy', 'ID', 'cluster']


def carregar_dados(caminho_csv):
    """
    Carrega o dataset CSV e mostra informações básicas.
    
    Parâmetros:
        caminho_csv: Caminho para o ficheiro CSV
    
    Retorna:
        DataFrame com os dados carregados
    """
    df = pd.read_csv(caminho_csv)
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def tratar_missing_values(df):
    """
    Preenche valores em falta usando a mediana de cada coluna.
    A mediana é mais robusta que a média face a outliers.
    
    Parâmetros:
        df: DataFrame original
    
    Retorna:
        DataFrame sem valores em falta
    """
    df_clean = df.copy()
    colunas_com_nulos = df_clean.columns[df_clean.isnull().any()].tolist()
    
    for col in colunas_com_nulos:
        if df_clean[col].dtype in ['float64', 'int64']:
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
            print(f"  Coluna '{col}' preenchida com mediana = {mediana:.4f}")
    
    print(f"  Nulos restantes: {df_clean.isnull().sum().sum()}")
    return df_clean


def normalizar_distribuicoes(df, colunas):
    """
    Aplica transformação Yeo-Johnson para aproximar as variáveis
    contínuas à distribuição normal (requisito do enunciado).
    
    Parâmetros:
        df: DataFrame limpo
        colunas: Lista de colunas contínuas a transformar
    
    Retorna:
        (df_transformado, power_transformer_fitted)
    """
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df[colunas] = pt.fit_transform(df[colunas])
    print(f"  {len(colunas)} colunas transformadas com Yeo-Johnson")
    return df, pt


def criar_features(df):
    """
    Cria novas features derivadas das existentes.
    Cada feature tem significado no contexto de cibersegurança.
    
    Parâmetros:
        df: DataFrame com as features originais
    
    Retorna:
        DataFrame com as novas features adicionadas
    """
    # Rácio de entropia (entropia do handshake vs conteúdo)
    df['entropy_ratio'] = df['e4_entropy_h'] / (df['e5_entropy_c'] + 1e-8)
    
    # Throughput: bytes por milissegundo
    df['bytes_per_ms'] = df['e2c_total_bytes'] / (df['e6b_flow_duration_ms'] + 1e-8)
    
    # Diferença de tamanho cliente-servidor (assimetria)
    df['client_server_diff'] = df['e2_client_size'] - df['e2_server_record_len']
    
    # Rácio registo/total (proporção do payload)
    df['record_to_total_ratio'] = df['e2_client_record_len'] / (df['e2c_total_bytes'] + 1e-8)
    
    # Flag: protocolo TLS antigo (vulnerável)
    df['is_old_tls'] = (
        df['e1b_tls_version_SSL3.0'].astype(int) |
        df['e1b_tls_version_TLS1.0'].astype(int) |
        df['e2b_tls_version_SSL3.0'].astype(int) |
        df['e2b_tls_version_TLS1.0'].astype(int)
    ).astype(int)
    
    # Interação entre tempo e entropia
    df['time_entropy_interaction'] = df['e6_time_char'] * df['e4_entropy_h']
    
    print(f"  6 novas features criadas. Total: {df.shape[1]} colunas")
    return df


def preparar_dados(df, test_size=0.2):
    """
    Prepara os dados para treino:
    1. Separa features (X) e target (y)
    2. Divide em treino/teste
    3. Aplica SMOTE ao conjunto de treino
    4. Escala os dados
    
    Parâmetros:
        df: DataFrame processado
        test_size: Proporção do conjunto de teste (default: 20%)
    
    Retorna:
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # Separar features e target
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in COLUNAS_META if c in df.columns], errors='ignore'
    )
    y = df['label']
    feature_names = X.columns.tolist()
    
    # Dividir treino/teste (com estratificação)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Aplicar SMOTE ao treino
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE: {len(y_train)} -> {len(y_train_bal)} amostras de treino")
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler, feature_names


def treinar_e_avaliar(X_train, X_test, y_train, y_test):
    """
    Treina todos os modelos exigidos pelo enunciado e
    recolhe as métricas: Accuracy, Accuracy CV, F-Score, Time.
    
    Parâmetros:
        X_train, X_test: Dados de treino e teste (já escalados)
        y_train, y_test: Labels de treino e teste
    
    Retorna:
        (resultados_df, modelos_treinados)
    """
    # Definir todos os modelos exigidos
    modelos = {
        'K-Neighbors Classifier': KNeighborsClassifier(n_neighbors=5),
        'Linear Regression': LinearRegression(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    resultados = []
    
    for nome, modelo in modelos.items():
        # Treino
        inicio = time.time()
        modelo.fit(X_train, y_train)
        
        # Previsão
        if nome == 'Linear Regression':
            y_pred = (modelo.predict(X_test) >= 0.5).astype(int)
        else:
            y_pred = modelo.predict(X_test)
        
        tempo = time.time() - inicio
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-Validation
        if nome == 'Linear Regression':
            cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
            acc_cv = max(cv.mean(), 0)
        else:
            cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
            acc_cv = cv.mean()
        
        resultados.append({
            'Algoritmo': nome,
            'Accuracy': round(acc, 4),
            'Accuracy CV': round(acc_cv, 4),
            'F-Score': round(f1, 4),
            'Time (s)': round(tempo, 2)
        })
        
        print(f"  {nome}: Acc={acc:.4f} | CV={acc_cv:.4f} | F1={f1:.4f} | {tempo:.2f}s")
    
    df_resultados = pd.DataFrame(resultados).set_index('Algoritmo')
    df_resultados = df_resultados.sort_values('F-Score', ascending=False)
    
    return df_resultados, modelos


def guardar_modelos(modelos, scaler, features, pt, pasta='models'):
    """
    Guarda todos os modelos e artefactos em ficheiros .pkl.
    
    Parâmetros:
        modelos: Dicionário {nome: modelo_treinado}
        scaler: StandardScaler fitted
        features: Lista de nomes das features
        pt: PowerTransformer fitted
        pasta: Pasta de destino
    """
    os.makedirs(pasta, exist_ok=True)
    
    for nome, modelo in modelos.items():
        nome_ficheiro = nome.lower().replace(' ', '_').replace('-', '_')
        joblib.dump(modelo, os.path.join(pasta, f'{nome_ficheiro}.pkl'))
    
    joblib.dump(scaler, os.path.join(pasta, 'scaler.pkl'))
    joblib.dump(features, os.path.join(pasta, 'features_selecionadas.pkl'))
    joblib.dump(pt, os.path.join(pasta, 'power_transformer.pkl'))
    
    print(f"  {len(modelos)} modelos + artefactos guardados em '{pasta}/'")


# ============================================================
# EXECUÇÃO PRINCIPAL (quando executado diretamente)
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  PIPELINE COMPLETO — TREINO DE MODELOS")
    print("=" * 60)
    
    # 1. Carregar
    print("\n[1/6] Carregar dados...")
    df = carregar_dados('data/raw/ml_features_and_labels.csv')
    
    # 2. Tratar missing values
    print("\n[2/6] Tratar valores em falta...")
    df = tratar_missing_values(df)
    
    # 3. Normalizar distribuições
    print("\n[3/6] Aproximar à distribuição normal...")
    df, pt = normalizar_distribuicoes(df, COLUNAS_CONTINUAS)
    
    # 4. Feature engineering
    print("\n[4/6] Criar novas features...")
    df = criar_features(df)
    
    # 5. Preparar e treinar
    print("\n[5/6] Preparar dados e treinar modelos...")
    X_tr, X_te, y_tr, y_te, scaler, feat_names = preparar_dados(df)
    resultados, modelos = treinar_e_avaliar(X_tr, X_te, y_tr, y_te)
    
    # 6. Guardar
    print("\n[6/6] Guardar modelos...")
    guardar_modelos(modelos, scaler, feat_names, pt)
    
    # Mostrar tabela final
    print("\n" + "=" * 60)
    print("  TABELA DE RESULTADOS")
    print("=" * 60)
    print(resultados.to_string())
    print("=" * 60)
    melhor = resultados['F-Score'].idxmax()
    print(f"\nMelhor modelo: {melhor} (F1 = {resultados.loc[melhor, 'F-Score']})")
