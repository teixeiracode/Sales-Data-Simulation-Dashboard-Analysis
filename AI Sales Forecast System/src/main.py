import sys
import os

# Fix Windows console UTF-8 encoding for emoji support
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# 1. CONFIGURAÇÕES INICIAIS
# =========================

def criar_diretorios():
    """Cria diretórios necessários para o projeto"""
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

# =========================
# 2. GERAR DADOS MELHORADOS
# =========================

def gerar_dados(qtd=5000, seed=42):
    """
    Gera dados de vendas mais realistas com sazonalidade
    
    Args:
        qtd: Quantidade de registros
        seed: Semente para reproducibilidade
    
    Returns:
        DataFrame com dados gerados
    """
    random.seed(seed)
    np.random.seed(seed)
    
    produtos = {
        "Mouse": 50,
        "Teclado": 150,
        "Monitor": 900,
        "Notebook": 3000,
        "Headset": 200
    }
    
    # Pesos para distribuição mais realista
    pesos_produtos = [0.3, 0.25, 0.15, 0.1, 0.2]  # Mouse mais comum, Notebook menos
    
    cidades = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Salvador"]
    pesos_cidades = [0.4, 0.25, 0.15, 0.1, 0.1]  # SP mais vendas
    
    data_inicial = datetime(2024, 1, 1)
    
    dados = []
    
    for i in range(qtd):
        # Data com tendência de aumento nas vendas
        dias_offset = random.randint(0, 180)
        data = data_inicial + timedelta(days=dias_offset)
        
        # Sazonalidade: fim de semana tem mais vendas?
        is_weekend = data.weekday() >= 5
        multiplicador_qtd = 1.5 if is_weekend else 1.0
        
        produto = random.choices(list(produtos.keys()), weights=pesos_produtos)[0]
        cidade = random.choices(cidades, weights=pesos_cidades)[0]
        
        # Quantidade com distribuição mais realista (Poisson)
        quantidade = np.random.poisson(lam=3) + 1
        quantidade = int(quantidade * multiplicador_qtd)
        
        valor_unitario = produtos[produto]
        
        # Adicionar pequena variação de preço
        variacao = random.uniform(0.95, 1.05)
        valor_unitario = round(valor_unitario * variacao, 2)
        
        dados.append([
            data.strftime("%d/%m/%Y"),
            produto,
            cidade,
            quantidade,
            valor_unitario
        ])
    
    df = pd.DataFrame(dados, columns=[
        "data", "produto", "cidade", "quantidade", "valor_unitario"
    ])
    
    df.to_csv("data/vendas.csv", index=False)
    print(f"✅ CSV gerado com sucesso! ({qtd} registros)")
    
    return df

# =========================
# 3. ETL + ANÁLISE AVANÇADA
# =========================

def tratar_dados(df):
    """Tratamento de dados com validações"""
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["faturamento"] = df["quantidade"] * df["valor_unitario"]
    df["mes"] = df["data"].dt.month
    df["dia_semana"] = df["data"].dt.day_name()
    df["semana"] = df["data"].dt.isocalendar().week
    
    # Remover outliers (opcional)
    Q1 = df["faturamento"].quantile(0.25)
    Q3 = df["faturamento"].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = df[(df["faturamento"] < limite_inferior) | (df["faturamento"] > limite_superior)]
    if len(outliers) > 0:
        print(f"⚠️ {len(outliers)} outliers detectados e removidos")
        df = df[(df["faturamento"] >= limite_inferior) & (df["faturamento"] <= limite_superior)]
    
    return df

def analise_avancada(df):
    """Análise exploratória completa"""
    print("\n" + "="*50)
    print("📊 RELATÓRIO COMPLETO DE VENDAS")
    print("="*50)
    
    print(f"\n💰 FATURAMENTO TOTAL: R$ {df['faturamento'].sum():,.2f}")
    print(f"📦 TOTAL DE ITENS VENDIDOS: {df['quantidade'].sum():,}")
    print(f"📊 MÉDIA POR TRANSAÇÃO: R$ {df['faturamento'].mean():,.2f}")
    print(f"🎯 MEDIANA DO FATURAMENTO: R$ {df['faturamento'].median():,.2f}")
    
    # Análise por produto
    print("\n" + "-"*50)
    print("🏆 TOP PRODUTOS (por quantidade):")
    produto_vendas = df.groupby("produto")["quantidade"].sum().sort_values(ascending=False)
    for produto, qtd in produto_vendas.head(3).items():
        print(f"   {produto}: {qtd:,} unidades")
    
    # Análise por cidade
    print("\n🌎 TOP CIDADES (por faturamento):")
    cidade_faturamento = df.groupby("cidade")["faturamento"].sum().sort_values(ascending=False)
    for cidade, fat in cidade_faturamento.head(3).items():
        print(f"   {cidade}: R$ {fat:,.2f}")
    
    # Análise temporal
    print("\n📅 ANÁLISE TEMPORAL:")
    df["mes_nome"] = df["data"].dt.strftime("%B")
    vendas_por_mes = df.groupby("mes_nome")["faturamento"].sum()
    mes_top = vendas_por_mes.idxmax()
    print(f"   Mês com maior faturamento: {mes_top}")
    
    # Dia da semana com mais vendas
    dia_top = df.groupby("dia_semana")["faturamento"].sum().idxmax()
    print(f"   Dia da semana com mais vendas: {dia_top}")
    
    # Ticket médio por produto
    print("\n🎫 TICKET MÉDIO POR PRODUTO:")
    ticket_medio = df.groupby("produto").apply(
        lambda x: x["faturamento"].sum() / x["quantidade"].sum()
    ).sort_values(ascending=False)
    for produto, ticket in ticket_medio.head(3).items():
        print(f"   {produto}: R$ {ticket:.2f}")

# =========================
# 4. MODELO PREDITIVO MELHORADO
# =========================

def prever_avancado(df):
    """Previsão com validação e múltiplas features"""
    # Agregar por data
    df_group = df.groupby("data")["faturamento"].sum().reset_index()
    
    # Criar features temporais
    df_group["dia_semana"] = df_group["data"].dt.dayofweek
    df_group["mes"] = df_group["data"].dt.month
    df_group["dia_do_mes"] = df_group["data"].dt.day
    df_group["dias"] = np.arange(len(df_group))
    
    # Features defasadas (lag)
    for lag in [1, 2, 3]:
        df_group[f"lag_{lag}"] = df_group["faturamento"].shift(lag)
    
    # Remover linhas com NaN
    df_group = df_group.dropna()
    
    # Selecionar features
    feature_cols = ["dias", "dia_semana", "mes", "dia_do_mes", "lag_1", "lag_2", "lag_3"]
    X = df_group[feature_cols]
    y = df_group["faturamento"]
    
    # Dividir em treino e teste (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Treinar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Avaliar
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    print("\n" + "="*50)
    print("🤖 AVALIAÇÃO DO MODELO")
    print("="*50)
    print(f"R² (Treino): {r2_score(y_train, y_pred_train):.3f}")
    print(f"R² (Teste): {r2_score(y_test, y_pred_test):.3f}")
    print(f"MAE (Teste): R$ {mean_absolute_error(y_test, y_pred_test):.2f}")
    
    # Importância das features
    print("\n📊 IMPORTÂNCIA DAS FEATURES:")
    for feat, coef in zip(feature_cols, modelo.coef_):
        print(f"   {feat}: {coef:.2f}")
    
    # Previsão para próximos 7 dias
    ultima_data = df_group["data"].iloc[-1]
    previsoes = []
    
    for i in range(1, 8):
        nova_data = ultima_data + timedelta(days=i)
        
        # Criar features para o novo dia
        novo_dia = {
            "dias": len(df_group) + i,
            "dia_semana": nova_data.dayofweek,
            "mes": nova_data.month,
            "dia_do_mes": nova_data.day,
            "lag_1": y_test.iloc[-1] if len(y_test) > 0 else y_train.iloc[-1],
            "lag_2": y_test.iloc[-2] if len(y_test) > 1 else y_train.iloc[-2],
            "lag_3": y_test.iloc[-3] if len(y_test) > 2 else y_train.iloc[-3]
        }
        
        novo_X = np.array([[novo_dia[feat] for feat in feature_cols]])
        previsao = modelo.predict(novo_X)[0]
        previsoes.append(max(0, previsao))  # Não permitir valores negativos
    
    print("\n🔮 PREVISÃO DE FATURAMENTO (PRÓXIMOS 7 DIAS):")
    for i, valor in enumerate(previsoes, 1):
        data_previsao = ultima_data + timedelta(days=i)
        print(f"   {data_previsao.strftime('%d/%m/%Y')}: R$ {valor:,.2f}")
    
    return df_group, previsoes, modelo

# =========================
# 5. GRÁFICOS MELHORADOS
# =========================

def graficos_completos(df_group, previsoes, df_original):
    """Cria múltiplos gráficos para análise"""
    
    # Configuração do estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Figura 1: Evolução e Previsão
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Dashboard de Análise de Vendas", fontsize=16, fontweight='bold')
    
    # Gráfico 1: Série temporal com previsão
    ax1 = axes[0, 0]
    ax1.plot(df_group["dias"], df_group["faturamento"], label="Histórico", linewidth=2)
    ax1.plot(
        range(len(df_group), len(df_group) + len(previsoes)),
        previsoes,
        label="Previsão",
        linestyle='--',
        linewidth=2,
        color='red'
    )
    ax1.set_title("Evolução do Faturamento com Previsão", fontsize=12)
    ax1.set_xlabel("Dias")
    ax1.set_ylabel("Faturamento (R$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Top produtos
    ax2 = axes[0, 1]
    produto_vendas = df_original.groupby("produto")["faturamento"].sum().sort_values()
    produto_vendas.plot(kind="barh", ax=ax2, color='skyblue')
    ax2.set_title("Faturamento por Produto", fontsize=12)
    ax2.set_xlabel("Faturamento (R$)")
    
    # Gráfico 3: Vendas por cidade
    ax3 = axes[1, 0]
    cidade_vendas = df_original.groupby("cidade")["faturamento"].sum().sort_values()
    cidade_vendas.plot(kind="bar", ax=ax3, color='lightcoral')
    ax3.set_title("Faturamento por Cidade", fontsize=12)
    ax3.set_xlabel("Cidade")
    ax3.set_ylabel("Faturamento (R$)")
    ax3.tick_params(axis='x', rotation=45)
    
    # Gráfico 4: Vendas por dia da semana
    ax4 = axes[1, 1]
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    vendas_dia = df_original.groupby("dia_semana")["faturamento"].sum().reindex(dias_ordem)
    vendas_dia.plot(kind="bar", ax=ax4, color='lightgreen')
    ax4.set_title("Faturamento por Dia da Semana", fontsize=12)
    ax4.set_xlabel("Dia da Semana")
    ax4.set_ylabel("Faturamento (R$)")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("outputs/dashboard_vendas.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Gráficos salvos em 'outputs/dashboard_vendas.png'")

# =========================
# 6. EXPORTAR RELATÓRIO
# =========================

def exportar_relatorio(df, previsoes):
    """Exporta resultados para CSV e Excel"""
    
    # Resumo por produto
    resumo_produto = df.groupby("produto").agg({
        "quantidade": "sum",
        "faturamento": "sum",
        "valor_unitario": "mean"
    }).round(2)
    resumo_produto.columns = ["Total Vendido", "Faturamento Total", "Preço Médio"]
    resumo_produto.to_csv("outputs/resumo_produtos.csv")
    
    # Resumo por cidade
    resumo_cidade = df.groupby("cidade").agg({
        "quantidade": "sum",
        "faturamento": "sum"
    }).round(2)
    resumo_cidade.columns = ["Total Vendido", "Faturamento Total"]
    resumo_cidade.to_csv("outputs/resumo_cidades.csv")
    
    # Previsões
    previsoes_df = pd.DataFrame({
        "Dia": range(1, 8),
        "Previsao_Faturamento": previsoes
    })
    previsoes_df.to_csv("outputs/previsoes.csv", index=False)
    
    print("\n✅ Relatórios exportados para a pasta 'outputs/'")

# =========================
# 7. MAIN MELHORADA
# =========================

def main():
    """Função principal do pipeline"""
    print("🚀 Iniciando pipeline de análise de vendas...")
    
    # Criar diretórios
    criar_diretorios()
    
    # Gerar dados
    df = gerar_dados(5000, seed=42)
    
    # Tratar dados
    df = tratar_dados(df)
    
    # Análise
    analise_avancada(df)
    
    # Previsão
    df_group, previsoes, modelo = prever_avancado(df)
    
    # Gráficos
    graficos_completos(df_group, previsoes, df)
    
    # Exportar
    exportar_relatorio(df, previsoes)
    
    print("\n" + "="*50)
    print("✨ Pipeline concluído com sucesso!")
    print("="*50)

if __name__ == "__main__":
    main()