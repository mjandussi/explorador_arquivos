import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import squarify
from scipy.stats import binom
from scipy.stats import poisson

# Configura para usar o layout amplo
st.set_page_config(layout="wide")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: visible;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

html_temp = """
<div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Explorador de Arquivos</p></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.subheader("Análise Estatística de Arquivos CSV com Streamlit")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

# Função para carregar e processar o DataFrame
def ler_data(file):
    # Tentar ler o arquivo CSV com diferentes codificações
    encodings = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding, sep=';')
            st.success(f"Arquivo carregado com sucesso usando a codificação {encoding}!")
            return df
        except Exception as e:
            st.warning(f"Tentativa de carregar com codificação {encoding} falhou: {e}")
    st.error("Falha ao carregar o arquivo com as codificações testadas.")
    return None

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# Função para exibir e editar o DataFrame
def editar_dataframe():
    df = st.session_state.df
    if st.checkbox("Mostrar os dados"):
        number = st.number_input("Número de Linhas para Visualizar", min_value=1, max_value=len(df), value=5)
        st.dataframe(df.head(number))


    st.divider()

    # Mostrar Datatypes
    st.subheader("Verificar os tipos de dados das colunas")
    if st.checkbox("Tipos de dados"):
        st.write(df.dtypes)


    st.divider()

    # Seleção de colunas para alterar o tipo de dados para Float
    colunas_nao_float = df.select_dtypes(exclude=['float']).columns.tolist()
    st.subheader("Alterar Tipo de Dados para Float (número decimal)")
    columns_to_convert = st.multiselect("Selecione as colunas que deseja converter para número decimal", colunas_nao_float)
    
    if columns_to_convert:
        try:
            # Substituir vírgulas por pontos nas colunas selecionadas
            for col in columns_to_convert:
                df[col] = df[col].str.replace(',', '.').astype(float)
            st.success(f"Colunas {columns_to_convert} convertidas para float com sucesso!")
        except ValueError as e:
            st.error(f"Erro ao converter colunas: {e}")

    st.divider()

    # Seleção de colunas para alterar o tipo de dados para Texto
    colunas_nao_texto = df.select_dtypes(exclude=['object']).columns.tolist()
    st.subheader("Alterar Tipo de Dados para Texto")
    columns_to_convert_texto = st.multiselect("Selecione as colunas que deseja converter para texto", colunas_nao_texto)
    
    if columns_to_convert_texto:
        try:
            df[columns_to_convert_texto] = df[columns_to_convert_texto].astype(str)
            st.success(f"Colunas {columns_to_convert_texto} convertidas para texto com sucesso!")
        except ValueError as e:
            st.error(f"Erro ao converter colunas: {e}")


    st.divider()

    # Seleção de colunas para alterar o tipo de dados para Inteiro
    colunas_nao_inteiro = df.select_dtypes(exclude=['int']).columns.tolist()
    st.subheader("Alterar Tipo de Dados para Inteiro (número inteiro)")
    columns_to_convert_inteiro = st.multiselect("Selecione as colunas que deseja converter para número inteiro", colunas_nao_inteiro)
    
    if columns_to_convert_inteiro:
        try:
            df[columns_to_convert_inteiro] = df[columns_to_convert_inteiro].astype(int)
            st.success(f"Colunas {columns_to_convert_inteiro} convertidas para inteiro com sucesso!")
        except ValueError as e:
            st.error(f"Erro ao converter colunas: {e}")

    st.divider()
    
    # Seleção de colunas para alterar o tipo de dados para Data
    colunas_nao_data = df.select_dtypes(exclude=['datetime']).columns.tolist()
    st.subheader("Alterar Tipo de Dados para Data")
    columns_to_convert_data = st.multiselect("Selecione as colunas que deseja converter para data", colunas_nao_data)
    
    if columns_to_convert_data:
        try:
            df[columns_to_convert_data] = df[columns_to_convert_data].apply(pd.to_datetime)
            st.success(f"Colunas {columns_to_convert_data} convertidas para data com sucesso!")
        except Exception as e:
            st.error(f"Erro ao converter colunas: {e}")
    

    st.divider()

    st.subheader("Extrair os Primeiros Dígitos de uma Coluna")
    col_to_extract = st.selectbox("Selecione a Coluna para Extrair os Primeiros Dígitos", df.columns.tolist())
    new_col_extract_name = st.text_input("Nome da Nova Coluna")
    cut_value = st.number_input("Valor de Corte para os Primeiros Dígitos", value=2)
    
    if st.button("Extrair Primeiros Dígitos"):
        try:
            df[new_col_extract_name] = df[col_to_extract].astype(str).str.slice(0, cut_value)
            st.success(f"Nova coluna '{new_col_extract_name}' criada com os primeiros dígitos de '{col_to_extract}'!")
        except Exception as e:
            st.error(f"Erro ao extrair os primeiros dígitos: {e}")


    st.divider()

    st.subheader("Extrair os Últimos Dígitos de uma Coluna")
    col_to_extract_last = st.selectbox("Selecione a Coluna para Extrair os Últimos Dígitos", df.columns.tolist(), key="last_digits_col")
    new_col_extract_last_name = st.text_input("Nome da Nova Coluna para os Últimos Dígitos", key="last_digits_col_name")
    cut_value_last = st.number_input("Valor de Corte para os Últimos Dígitos", value=2, key="last_digits_cut_value")
    
    if st.button("Extrair Últimos Dígitos"):
        try:
            df[new_col_extract_last_name] = df[col_to_extract_last].astype(str).str.slice(-cut_value_last)
            st.success(f"Nova coluna '{new_col_extract_last_name}' criada com os últimos dígitos de '{col_to_extract_last}'!")
        except Exception as e:
            st.error(f"Erro ao extrair os últimos dígitos: {e}")

    
    
    # Atualiza o DataFrame no session_state
    st.session_state.df = df

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# Função para exibir gráficos e análises estatísticas
def exibir_graficos():
    df = st.session_state.df

     # Criando o DF de apenas colunas numericas
    df_numericas = df.select_dtypes(include=['number'])
    df_numericas_a =  df_numericas.copy()


    # Exibir informações adicionais sobre o DataFrame
    st.subheader("Número de Linhas e Colunas:")
    if st.checkbox("Número de Linhas e Colunas"):
        st.write(df.shape)

    st.divider()

    st.subheader("Resumo Estatístico:")
    if st.checkbox("Resumo Estatístico"):
        st.write(df.describe())
    
    st.divider()

    # Cálculo de Métricas Estatísticas
    st.subheader("Métricas Estatísticas")
    if st.checkbox("Mostrar Métricas Estatísticas", key="show_metrics"):
        for coluna in df_numericas_a.columns:
            media = df_numericas_a[coluna].mean()
            mediana = df_numericas_a[coluna].median()
            desvio_padrao = df_numericas_a[coluna].std()
            coef_variação = desvio_padrao / media if media != 0 else np.nan

            Q1 = df_numericas_a[coluna].quantile(0.25)
            Q3 = df_numericas_a[coluna].quantile(0.75)
            IQR = Q3 - Q1

            df_numericas_a.loc['média', coluna] = media
            df_numericas_a.loc['mediana', coluna] = mediana
            df_numericas_a.loc['desvio padrão', coluna] = desvio_padrao
            df_numericas_a.loc['coeficiente de variação', coluna] = coef_variação
            df_numericas_a.loc['distância interquartílica', coluna] = IQR

        st.write("Métricas Estatísticas: 'média', 'mediana', 'desvio padrão', 'coeficiente de variação e distância interquartílica")
        st.dataframe(df_numericas_a.tail(5))

    st.divider()

    st.subheader("Análise de Contagem de Valores por Coluna")
    selected_columns = st.multiselect("Selecione a Coluna", df.columns.tolist())
    if selected_columns:
        new_df = df[selected_columns]
        st.write(new_df.value_counts())
    else:
        st.warning("Nenhuma coluna selecionada.")




    st.divider()

    st.subheader("Gráfico de Histograma e Cálculo do Coeficiente de Assimetria (Skewness)")
    selected_hist_column = st.selectbox("Selecione a coluna", df_numericas.columns.tolist(), key="skew_column")
    
    if st.button("Calcular Assimetria, Curtose e Plotar o Histograma", key="calculate_skewness_and_plot"):
        texto_skewness = """
        Skewness = 0: Indica uma distribuição simétrica.
        Skewness > 0: Indica uma assimetria positiva, onde a cauda da distribuição é mais longa para o lado direito da curva.
        Skewness < 0: Indica uma assimetria negativa, onde a cauda da distribuição é mais longa para o lado esquerdo da curva.
        """
        st.text(texto_skewness)
        skewness = df_numericas[selected_hist_column].skew()
        st.success(f"Coeficiente de Assimetria (Skewness) da coluna '{selected_hist_column}': {skewness}")
        texto_curtose = """
        Curtose = 0: Indica uma distribuição com o mesmo achatamento que a distribuição normal.
        Curtose > 0: Indica uma distribuição mais "leptocúrtica" com caudas mais pesadas e um pico mais agudo do que a distribuição normal. Isso sugere uma maior probabilidade de outliers.
        Curtose < 0: Indica uma distribuição mais "platicúrtica" com caudas mais leves e um achatamento maior do que a distribuição normal.
        """
        st.text(texto_curtose)
        curtose = df_numericas[selected_hist_column].kurtosis()
        st.success(f"Coeficiente de Curtose da coluna '{selected_hist_column}': {curtose}")
        
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_numericas[selected_hist_column], kde=True, color="red", bins=7, ax=ax)
        
        plt.title(f'Distribuição da {selected_hist_column}', fontsize=20)
        plt.xlabel(selected_hist_column, fontsize=14)
        plt.ylabel('Frequência', fontsize=14)
        
        st.pyplot(fig)


    st.divider()

    st.subheader("Gráfico de Setores (Pizza)")
    selected_pie_column = st.selectbox("Selecione a coluna para o gráfico de Setores (Pizza)", df.columns.tolist(), key="pie_column")
    
    if st.button("Gerar Gráfico de Setores (Pizza)", key="generate_pie_chart"):
        st.success(f"Gerando gráfico de pizza para a coluna {selected_pie_column}")
        
        if df[selected_pie_column].dtype == 'object' or df[selected_pie_column].nunique() < 10:
            distribuicao = df[selected_pie_column].value_counts()
            
            fig, ax = plt.subplots()
            distribuicao.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['red', 'blue', 'pink', 'green', 'yellow', 'purple', 'grey', 'orange', 'black', 'white'], ax=ax)
            
            plt.title(f'Distribuição de {selected_pie_column}', fontsize=16)
            plt.ylabel('')
            
            st.pyplot(fig)
        else:
            st.error("Por favor, selecione uma coluna categórica ou com poucas categorias.")


    st.divider()
    df_numericas_2 = df_numericas.copy()
    st.subheader("Cálculo Dinâmico de Intervalos de Faixas e Gráfico de Barras")
    selected_bin_column = st.selectbox("Selecione a coluna para calcular as faixas", df_numericas_2.columns.tolist(), key="bin_column")
    
    if 'faixas' not in st.session_state:
        st.session_state['faixas'] = None
    if 'frequencia_faixas' not in st.session_state:
        st.session_state['frequencia_faixas'] = None

    step = st.number_input("Defina o intervalo de cada faixa para sua coluna", min_value=1, value=10)

    if st.button("Calcular Faixas", key="calculate_bins"):
        st.success(f"Calculando faixas para a coluna {selected_bin_column}")    
        min_val = df_numericas_2[selected_bin_column].min()
        max_val = df_numericas_2[selected_bin_column].max()
        faixas = list(range(int(min_val), int(max_val) + step, step))
        
        faixa_col_name = f"Faixa_{selected_bin_column}"
        df_numericas_2[faixa_col_name] = pd.cut(df_numericas_2[selected_bin_column], bins=faixas, right=False, include_lowest=True)
        
        frequencia_faixas = df_numericas_2[faixa_col_name].value_counts().sort_index()
        
        st.session_state['faixas'] = faixas
        st.session_state['frequencia_faixas'] = frequencia_faixas

    if st.session_state['frequencia_faixas'] is not None:
        st.write(st.session_state['frequencia_faixas'])
        
        st.session_state['frequencia_faixas'].index = st.session_state['frequencia_faixas'].index.astype(str)

        fig, ax = plt.subplots(figsize=(10, 6))

        cores = ['#1f77b4' if (barra != max(st.session_state['frequencia_faixas'].values)) else '#d62728' for barra in st.session_state['frequencia_faixas'].values]

        barras = ax.bar(st.session_state['frequencia_faixas'].index, st.session_state['frequencia_faixas'].values, width=0.8, color=cores, edgecolor='grey')

        ax.set_title(f'Frequência por Faixa - {selected_bin_column}', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Faixa - {selected_bin_column}', fontsize=14)
        ax.set_ylabel('Frequência', fontsize=14)

        plt.xticks(rotation=45, ha="right")

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

        for bar in barras:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)



    st.divider()

    df2 = df.copy()
    st.subheader("Tabela de Frequência Cruzada")
    cola, colb = st.columns(2)
    
    with cola:
        col1 = st.selectbox("Selecione a primeira coluna", df2.columns.tolist(), key="cross_tab_col1")
    
    with colb:
        col2 = st.selectbox("Selecione a segunda coluna", df2.columns.tolist(), key="cross_tab_col2")
    
    if st.button("Gerar Tabela de Frequência Cruzada", key="generate_cross_tab"):
        st.success(f"Gerando tabela de frequência cruzada para {col1} e {col2}")
        st.write("Proporção com Totais")
        tabela_frequencias = pd.crosstab(df2[col1], df2[col2], margins=True, margins_name="Total")
        st.write(tabela_frequencias)
        st.divider()
        st.write("Proporção Percentual")
        tabela_proporcao = pd.crosstab(df2[col1], df2[col2], normalize='all')
        st.write(tabela_proporcao)
        st.divider()
        st.write("Proporção Percentual por linhas")
        tabela_proporcao_linhas = pd.crosstab(df2[col1], df2[col2], normalize='index')
        st.write(tabela_proporcao_linhas)


    st.divider()

    st.subheader("Boxplot Personalizado")
    selected_boxplot_column = st.selectbox("Selecione a coluna para o boxplot", df_numericas.columns.tolist(), key="boxplot_column")
    
    if st.button("Gerar Boxplot", key="generate_boxplot"):
        st.success(f"Gerando boxplot para a coluna {selected_boxplot_column}")
        
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_numericas[selected_boxplot_column], orient="h", color="lightcoral", ax=ax)
        
        mean_value = df_numericas[selected_boxplot_column].mean()
        ax.axvline(mean_value, color='red', linestyle='--', linewidth=2.5, label=f'Média: {mean_value:.2f}')
        
        plt.title(f'Boxplot da {selected_boxplot_column}', fontsize=20)
        plt.xlabel(selected_boxplot_column, fontsize=14)
        plt.ylabel('Variáveis', fontsize=14)
        
        plt.legend()
        
        st.pyplot(fig)


    st.divider()
    df_numericas3 = df_numericas.copy()
    st.subheader("Gráfico de Dispersão com Linha de Tendência")
    colc, cold = st.columns(2)
    
    with colc:
        x_col = st.selectbox("Selecione a coluna para o eixo X", df_numericas3.columns.tolist(), key="scatter_x_col")
    
    with cold:
        y_col = st.selectbox("Selecione a coluna para o eixo Y", df_numericas3.columns.tolist(), key="scatter_y_col")
    
    if st.button("Gerar Gráfico de Dispersão", key="generate_scatter_plot"):
        st.success(f"Gerando gráfico de dispersão para {x_col} e {y_col}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=x_col, y=y_col, data=df_numericas3, scatter_kws={'s': 50}, line_kws={"color": "red"}, ax=ax)
        
        plt.title(f'Diagrama de Dispersão entre {x_col} e {y_col} com Linha de Tendência')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        st.pyplot(fig)


    st.divider()

    st.subheader("Análise Bivariada - Boxplot (variável quantitativa / variável qualitativa)")

    # Seleção das colunas para os gráficos

    df1 = df.copy()
    coluna_x1 = st.selectbox("Selecione a coluna X para o primeiro gráfico", df1.columns.tolist(), key="bivariada_x1")
    coluna_y1 = st.selectbox("Selecione a coluna Y para o primeiro gráfico", df1.columns.tolist(), key="bivariada_y1")
    coluna_x2 = st.selectbox("Selecione a coluna X para o segundo gráfico", df1.columns.tolist(), key="bivariada_x2")
    coluna_y2 = st.selectbox("Selecione a coluna Y para o segundo gráfico", df1.columns.tolist(), key="bivariada_y2")

    # Verificar se as colunas foram selecionadas
    if coluna_x1 and coluna_y1 and coluna_x2 and coluna_y2:
        if st.button("Gerar Análise", key="generate_analise"):
            st.success("Gerando boxplot para a análise")
            # Configurando o layout para 1 linha e 2 colunas de gráficos
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Primeiro boxplot
            sns.boxplot(x=coluna_x1, y=coluna_y1, data=df1, ax=axes[0])
            axes[0].set_title(f'Distribuição de {coluna_y1} por {coluna_x1}')
            axes[0].set_xlabel(coluna_x1)
            axes[0].set_ylabel(coluna_y1)

            # Segundo boxplot
            sns.boxplot(x=coluna_x2, y=coluna_y2, data=df1, ax=axes[1])
            axes[1].set_title(f'Distribuição de {coluna_y2} por {coluna_x2}')
            axes[1].set_xlabel(coluna_x2)
            axes[1].set_ylabel(coluna_y2)

            # Ajusta automaticamente os subplots para que caibam no layout
            plt.tight_layout()

            # Exibir os gráficos no Streamlit
            st.pyplot(fig)


    st.divider()

    st.subheader("Heatmap")
    colunas_selecionadas = []
    colunas_selecionadas  = st.multiselect("Colunas", options=df_numericas.columns.tolist())
            
    # Verifica se há colunas selecionadas
    if colunas_selecionadas:
        df_selecionado = df_numericas[colunas_selecionadas]

        # Gera o heatmap
        st.header("Mapa de Calor das Correlações")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_selecionado.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        st.pyplot(plt.gcf())
    else:
        st.warning("Por favor, selecione pelo menos uma coluna para gerar o heatmap.")


    st.divider()

    st.subheader("Treemap")

    # Filtrando colunas de texto
    colunas_texto = df.select_dtypes(include=['object']).columns.tolist()

    # Selecionando as colunas para os valores e rótulos
    df_numericas = df.select_dtypes(include=['number'])
    s_size_col = st.selectbox("Selecione a coluna para os valores", df_numericas.columns.tolist(), key="treemap_1")
    s_label_col = st.selectbox("Selecione a coluna para as características", colunas_texto, key="treemap_2")
    s_color_col = st.selectbox("Selecione a coluna para as cores", colunas_texto, key="treemap_3")

    # Checkbox para agrupar e somar os valores
    resumir = st.checkbox("Deseja agrupar e somar os valores pelas colunas escolhidas (resumir os dados)?", key="resumir")

    if s_size_col and s_label_col and s_color_col:
        if st.button("Gerar Análise", key="generate_treemap"):
            st.success("Gerando Treemap")
            
            # Obtendo os dados das colunas selecionadas
            s_size = df[s_size_col]
            s_label = df[s_label_col]
            s_color = df[s_color_col]
            
            # Agrupando e somando os valores se o checkbox estiver marcado
            if resumir:
                df_resumido = df.groupby([s_label_col, s_color_col])[s_size_col].sum().reset_index()
                s_size = df_resumido[s_size_col].tolist()
                s_label = df_resumido[s_label_col].tolist()
                s_color = df_resumido[s_color_col].tolist()
            else:
                s_size = s_size.tolist()
                s_label = s_label.tolist()
                s_color = s_color.tolist()
            
            # Verificando se a soma dos tamanhos é zero
            if sum(s_size) == 0:
                st.error("A soma dos valores selecionados é zero. Por favor, selecione uma coluna com valores maiores que zero.")
            else:
                # Criando a figura
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Criando um dicionário de cores para as categorias
                unique_colors = list(set(s_color))
                color_dict = {category: plt.cm.tab20(i / len(unique_colors)) for i, category in enumerate(unique_colors)}
                colors = [color_dict[category] for category in s_color]
                
                # Criando o gráfico de treemap
                squarify.plot(sizes=s_size, label=s_label, alpha=.8, ax=ax, color=colors, text_kwargs={'fontsize':6})

                # Adicionando título
                ax.set_title(f'Distribuição de {s_size_col} por {s_label_col} com cores baseadas em {s_color_col}')
                
                # Criando a legenda
                handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[category]) for category in unique_colors]
                ax.legend(handles, unique_colors, loc='upper left', bbox_to_anchor=(1, 1))        

                # Removendo os eixos
                ax.axis('off')

                # Exibindo o gráfico no Streamlit
                st.pyplot(fig)



    
    st.divider()
    if st.button("Obrigado!!"):
            st.balloons()





def modelos_probabilisticos():
    st.header("Testes de Modelos Probabilísticos")
    

    # Título do aplicativo
    st.title("Cálculo de Probabilidades da Distribuição Binomial")

    # Entrada de parâmetros da distribuição binomial
    n = st.number_input("Número de amostras (n)", min_value=1, value=100, step=1)
    p = st.number_input("Probabilidade (p) em número decimal", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    k = st.number_input("Casos ocorridos (k)", min_value=0, value=1, step=1)

    # Probabilidade de (k) casos ocorrerem
    prob = binom.pmf(k, n, p)

    # Exibindo os resultados
    st.subheader("Resultados")
    if st.button("Cálculo de Probabilidades da Distribuição Binomial", key="generate_binominal"):
            st.success("Gerando o cálculo")
            st.write(f"Probabilidade de (k) casos ocorrerem é: {prob:.2%}")


    st.divider()

    # Título do aplicativo
    st.title("Cálculo de Probabilidade da Distribuição de Poisson")

    # Entrada de parâmetros da distribuição de Poisson
    lambda_param = st.number_input("Taxa média de ocorreência(λ)", min_value=0.0, value=10.0, step=0.1)
    k = st.number_input("Número de casos (k)", min_value=0, value=5, step=1)

    # Calculando a probabilidade usando a função de massa de probabilidade (PMF) da distribuição de Poisson
    probabilidade = poisson.pmf(k, lambda_param)

    # Exibindo o resultado
    st.subheader("Resultado")
    if st.button("Cálculo de Probabilidades da Distribuição de Poisson", key="generate_poisson"):
            st.success("Gerando o cálculo")
            st.write(f"Probabilidade de ocorrência: {probabilidade:.2%}")
    





###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# Navegação entre páginas
page = st.sidebar.radio("PÁGINAS", ["Edição do Arquivo", "Análises Estatísticas do Arquivo", "Testes de Modelos Probabilísticos"])

# Carregar o DataFrame
if uploaded_file is not None:
    if 'df' not in st.session_state:
        st.session_state.df = ler_data(uploaded_file)
    
    if st.session_state.df is not None:
        if page == "Edição do Arquivo":
            editar_dataframe()
        elif page == "Análises Estatísticas e Gráficos":
            exibir_graficos()
        # elif page == "Modelos Probabilísticos":
        #     modelos_probabilisticos()

if uploaded_file is None:
    if page == "Modelos Probabilísticos":
        modelos_probabilisticos()

  
else:
    st.info("Por favor, carregue um arquivo CSV para começar.")


st.sidebar.divider()

st.sidebar.header("Contato")
st.sidebar.info("mjandussi@gmail.com")
st.sidebar.text("Mantido por Marcelo Jandussi")