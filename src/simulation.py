# ============================================================
# simulation.py — Simulação do Uso dos Modelos Treinados
# ============================================================
# Este ficheiro simula o uso dos modelos "em produção":
# recebe dados de uma nova sessão de rede e classifica-a
# como Normal ou Anomalia usando os modelos guardados.
#
# Executar:
#   python src/simulation.py
#
# Pré-requisito: Correr primeiro o notebook 02_Full_Pipeline
# ou o models.py para gerar os modelos em models/
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os


def carregar_artefactos(pasta_modelos='models'):
    """
    Carrega todos os modelos treinados e artefactos necessários
    para fazer previsões em novos dados.
    
    Parâmetros:
        pasta_modelos: Pasta onde estão os ficheiros .pkl
    
    Retorna:
        (modelos, scaler, features, power_transformer)
    """
    # Lista de modelos guardados
    nomes_modelos = {
        'K-Neighbors Classifier': 'k_neighbors_classifier.pkl',
        'Linear Regression': 'linear_regression.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
    }
    
    modelos = {}
    for nome, ficheiro in nomes_modelos.items():
        caminho = os.path.join(pasta_modelos, ficheiro)
        if os.path.exists(caminho):
            modelos[nome] = joblib.load(caminho)
            print(f"  Modelo carregado: {nome}")
        else:
            print(f"  AVISO: Modelo '{nome}' nao encontrado em {caminho}")
    
    # Carregar artefactos de pré-processamento
    scaler = joblib.load(os.path.join(pasta_modelos, 'scaler.pkl'))
    features = joblib.load(os.path.join(pasta_modelos, 'features_selecionadas.pkl'))
    pt = joblib.load(os.path.join(pasta_modelos, 'power_transformer.pkl'))
    
    print(f"\n  {len(modelos)} modelos carregados com sucesso.")
    return modelos, scaler, features, pt


def criar_cenarios_simulacao(features):
    """
    Cria 3 cenários fictícios de tráfego de rede para demonstrar
    o uso dos modelos em situações reais.
    
    Cenário 1 — Tráfego Normal:
        Sessão TLS 1.3 típica com valores dentro da norma.
        
    Cenário 2 — Ataque (Exfiltração de Dados):
        Sessão com entropia muito alta, protocolo TLS antigo,
        e muitos bytes transferidos rapidamente.
        
    Cenário 3 — Caso Ambíguo:
        Sessão com algumas métricas anómalas mas outras normais.
        Testa a sensibilidade dos modelos.
    
    Parâmetros:
        features: Lista de nomes das features esperadas
    
    Retorna:
        Dicionário {nome_cenario: DataFrame_com_1_linha}
    """
    n_features = len(features)
    
    # Cenário 1: Valores "normais" (zeros = valores médios após StandardScaler)
    normal = np.zeros(n_features)
    
    # Cenário 2: Valores extremos que simulam um ataque
    ataque = np.zeros(n_features)
    for i, f in enumerate(features):
        if 'entropy' in f or 'bytes' in f:
            ataque[i] = 3.0   # 3 desvios padrão acima da média
        elif 'old_tls' in f:
            ataque[i] = 2.0   # Protocolo antigo ativo
        elif 'duration' in f or 'time' in f:
            ataque[i] = -2.0  # Sessão muito curta
    
    # Cenário 3: Valores ligeiramente anómalos
    ambiguo = np.zeros(n_features)
    for i, f in enumerate(features):
        if 'entropy' in f:
            ambiguo[i] = 1.0  # Ligeiramente elevado
    
    cenarios = {
        'Trafego Normal (TLS 1.3)': pd.DataFrame([normal], columns=features),
        'Ataque (Exfiltracao de Dados)': pd.DataFrame([ataque], columns=features),
        'Caso Ambiguo (Borderline)': pd.DataFrame([ambiguo], columns=features),
    }
    
    return cenarios


def classificar(modelos, scaler, dados):
    """
    Classifica um novo caso usando todos os modelos disponíveis.
    
    Parâmetros:
        modelos: Dicionário de modelos treinados
        scaler: StandardScaler fitted
        dados: DataFrame com 1 linha (novo caso)
    
    Retorna:
        DataFrame com as previsões de cada modelo
    """
    # Escalar os dados (usar o mesmo scaler do treino)
    dados_scaled = scaler.transform(dados)
    
    previsoes = {}
    for nome, modelo in modelos.items():
        if nome == 'Linear Regression':
            # Regressão linear dá valor contínuo — converter com threshold
            valor_raw = modelo.predict(dados_scaled)[0]
            pred = 1 if valor_raw >= 0.5 else 0
            previsoes[nome] = {
                'Previsao': 'ANOMALIA' if pred == 1 else 'NORMAL',
                'Valor_Bruto': round(valor_raw, 4)
            }
        else:
            pred = modelo.predict(dados_scaled)[0]
            previsoes[nome] = {
                'Previsao': 'ANOMALIA' if pred == 1 else 'NORMAL',
                'Valor_Bruto': int(pred)
            }
    
    return previsoes


def executar_simulacao():
    """
    Função principal que executa toda a simulação:
    1. Carrega os modelos guardados
    2. Cria cenários fictícios
    3. Classifica cada cenário com todos os modelos
    4. Apresenta os resultados
    """
    print("=" * 65)
    print("  SIMULACAO — USO DOS MODELOS PARA NOVOS CASOS")
    print("=" * 65)
    
    # 1. Carregar modelos
    print("\n[1] A carregar modelos treinados...")
    modelos, scaler, features, pt = carregar_artefactos()
    
    # 2. Criar cenários
    print("\n[2] A criar cenarios de simulacao...")
    cenarios = criar_cenarios_simulacao(features)
    print(f"  {len(cenarios)} cenarios criados.")
    
    # 3. Classificar cada cenário
    print("\n[3] A classificar novos casos...\n")
    print("=" * 65)
    
    for nome_cenario, dados in cenarios.items():
        print(f"\n  CENARIO: {nome_cenario}")
        print(f"  {'-' * 50}")
        
        previsoes = classificar(modelos, scaler, dados)
        
        for nome_modelo, resultado in previsoes.items():
            pred = resultado['Previsao']
            raw = resultado['Valor_Bruto']
            # Alinhar o output para ficar legível
            print(f"    {nome_modelo:<25} -> {pred:<10} (raw: {raw})")
        
        # Votação por maioria (ensemble simples)
        votos = [r['Previsao'] for r in previsoes.values()]
        voto_normal = votos.count('NORMAL')
        voto_anomalia = votos.count('ANOMALIA')
        decisao = 'ANOMALIA' if voto_anomalia > voto_normal else 'NORMAL'
        print(f"\n    >> DECISAO POR MAIORIA: {decisao} "
              f"({voto_anomalia} anomalia vs {voto_normal} normal)")
    
    print(f"\n{'=' * 65}")
    print("  Simulacao concluida com sucesso.")
    print("  Os modelos estao operacionais para uso em producao.")
    print(f"{'=' * 65}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == '__main__':
    executar_simulacao()
