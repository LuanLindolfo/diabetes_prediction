import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Configurações Iniciais da Página
st.set_page_config(
    page_title="Dashboard Diagnóstico de Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Sistema de Apoio ao Diagnóstico de Diabetes")
st.markdown("Dashboard de análise de modelos preditivos e simulador clínico baseado em dados de indicadores de saúde (BRFSS 2015).")
st.divider()

# 2. Dados estáticos de resultado (Extraídos do seu Jupyter Notebook)
dados_resultados = {
    "Modelo": ["SVM (Linear)", "Random Forest", "Regressão Logística", "KNN", "Naïve Bayes", "Árvore de Decisão"],
    "Acurácia": [0.7483, 0.7368, 0.7484, 0.7102, 0.7180, 0.6552],
    "Recall": [0.7923, 0.7778, 0.7698, 0.7383, 0.7112, 0.6536],
    "Precisão": [0.7272, 0.7179, 0.7372, 0.6979, 0.7198, 0.6544]
}
df_resultados = pd.DataFrame(dados_resultados)

# 3. Criação de Abas para organizar a interface
tab1, tab2 = st.tabs(["📊 Visão Geral dos Resultados", "🔮 Simulador de Diagnóstico"])

# ==========================================
# ABA 1: VISÃO GERAL DOS RESULTADOS
# ==========================================
with tab1:
    st.header("🏆 Ranking de Modelos de Machine Learning")
    st.markdown("Comparação dos modelos testados com foco na métrica de **Recall** (capacidade de não deixar passar nenhum caso positivo da doença).")
    
    # Criando colunas para organizar os cards de métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="🥇 Modelo Vencedor", value="SVM (Linear)", delta="Melhor Detecção")
    with col2:
        st.metric(label="🎯 Recall Máximo Alcançado", value="79.23%", delta="+1.45% vs 2º lugar")
    with col3:
        st.metric(label="📊 Acurácia Geral do Modelo", value="74.83%")

    st.write("---")
    
    # Dividindo a tela para Tabela e Gráfico
    col_tabela, col_grafico = st.columns([1.2, 1.8])
    
    with col_tabela:
        st.subheader("Tabela de Métricas")
        # Mostrando o dataframe com destaque (usando recurso nativo do Streamlit)
        st.dataframe(
            df_resultados.style.highlight_max(subset=['Recall'], color='lightgreen'),
            hide_index=True,
            use_container_width=True
        )
        
    with col_grafico:
        st.subheader("Comparativo de Recall por Modelo")
        # Preparando dados para o gráfico de barras
        df_chart = df_resultados[['Modelo', 'Recall']].set_index('Modelo').sort_values(by="Recall")
        st.bar_chart(df_chart, height=300)


# ==========================================
# ABA 2: SIMULADOR (DEPLOY DO MODELO .PKL)
# ==========================================
with tab2:
    st.header("🔬 Simulador Clínico")
    st.markdown("Insira as informações de saúde do paciente para obter uma predição usando o modelo **SVM (Linear)**.")
    
    # Tentativa de carregar o arquivo .pkl gerado no Notebook
    try:
        with open('modelo_diabetes_producao.pkl', 'rb') as arquivo:
            pacote = pickle.load(arquivo)
            
        modelo = pacote['modelo']
        scaler = pacote['scaler']
        features = pacote['features']
        
        st.success("✅ Modelo carregado com sucesso! Preencha os dados abaixo.")
        
        # Como o dataset original tem 21 colunas (features), gerar um formulário dinâmico
        with st.form("form_previsao"):
            st.subheader("Indicadores de Saúde")
            
            # Divide os inputs do formulário em 3 colunas para ficar bonito e compacto na tela
            cols = st.columns(3)
            valores_entrada = []
            
            for i, feature in enumerate(features):
                coluna_atual = cols[i % 3]
                with coluna_atual:
                    # Cria um number_input genérico para cada feature
                    # Na vida real, você poderia customizar (ex: idade = st.slider, fumante = st.selectbox)
                    valor = st.number_input(f"Valor para {feature}", value=0.0)
                    valores_entrada.append(valor)
            
            # Botão grande para enviar os dados
            botao_prever = st.form_submit_button("Gerar Previsão Diagnóstica", type="primary")
            
            if botao_prever:
                with st.spinner("Processando dados e consultando modelo..."):
                    # 1. Transforma a lista de inputs em array e redimensiona
                    X_novo = np.array(valores_entrada).reshape(1, -1)
                    
                    # 2. Padroniza (Scala) usando o scaler exportado do notebook
                    X_novo_padronizado = scaler.transform(X_novo)
                    
                    # 3. Faz a previsão
                    previsao = modelo.predict(X_novo_padronizado)[0]
                    
                    st.write("---")
                    if previsao == 1:
                        st.error("🚨 **Atenção:** O modelo identificou um alto risco/indicadores compatíveis com DIABETES.")
                        st.snow() # Opcional: efeito visual nativo do streamlit 
                    else:
                        st.success("🟢 **Resultado:** O paciente apresenta indicadores compatíveis com o grupo SAUDÁVEL.")
                        st.balloons() # Opcional: efeito comemorativo nativo do streamlit

    except FileNotFoundError:
        st.warning("⚠️ Arquivo `modelo_diabetes_producao.pkl` não encontrado. Certifique-se de executar o último bloco do seu notebook para exportar o modelo e de rodar o Streamlit na mesma pasta.")
