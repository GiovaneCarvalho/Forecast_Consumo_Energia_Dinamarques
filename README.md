# ⚡ Forecast de Consumo de Energia - Copenhague, Dinamarca

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge&logo=xgboost)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA/SARIMA-red?style=for-the-badge)

## 📌 Overview
O consumo de energia é um dos mais fortes indicadores do desenvolvimento e potencial econômico de um país. A previsão da demanda energética permite que indústrias, governos e distribuidoras ajam de forma proativa para alocar recursos, evitar apagões e baratear os custos de operação.

Neste projeto de portfólio, aplicamos metodologias de **Mineração de Dados** e **Modelagem de Séries Temporais** para prever o consumo diário de energia elétrica do município de Copenhague (København), capital da Dinamarca.

---

## 🎯 Objetivos do Projeto
Desenvolver um modelo de alta precisão para prever os dados de consumo de energia num horizonte futuro de 60 dias, utilizando técnicas que vão desde modelos clássicos de estatística (Box-Jenkins) até algoritmos robustos de *Machine Learning*.

**Principais tópicos abordados:**
- **Engenharia de Dados:** Coleta de dados via API e tratamento de _timestamps_.
- **Análise Exploratória (EDA):** Diferentes agregações, janelas deslizantes e análise de sazonalidade.
- **Séries Temporais:** Avaliação de estacionariedade (Teste ADF), diferenciação e decomposição.
- **Modelagem Clássica:** Correlogramas (ACF e PACF), modelos ARIMA e SARIMA.
- **Machine Learning:** 
  - Engenharia de features temporais (*day, week, day of year*) e *lags* temporais.
  - Modelos Ridge, Random Forest e XGBoost Regressor.
  - Tuner de hiperparâmetros e *Cross Validation* para séries temporais (`TimeSeriesSplit`).
- **Tracking de Experimentos:** Gerenciamento dos modelos e registro das métricas (MAE e MAPE) utilizando **MLflow**.

---

## 📊 Fonte de Dados e EDA
Os dados foram coletados publicamente através da [API da Energinet (Dinamarca)](https://en.energinet.dk/energy-data/data-catalog/). A base possui alta granularidade (horas) e segrega o consumo em três grandes áreas: **Indústria (Erhverv)**, **Público (Offentligt)** e **Privado (Privat)**.

### Descobertas da Avaliação Gráfica e Sazonalidade:
1. **Perfis de Consumo Distintos:** O setor **Industrial** consumia volumes massivamente superiores e de forma muito mais constante que os demais. O setor público era focado em volumes marginais e o privado apresentava forte assimetria durante o dia.
2. **Sazionalidade Diária e Semanal:** Agregando para _dias_, notamos fortes picos de consumo provindos do setor _Privado_ aos finais de semana e reduções expressivas na Indústria.
3. **Tendência Anual e Janelas Deslizantes:** Usando médias móveis (7 e 90 dias), identificou-se uma queda expressiva e repetida do consumo nos meses de veraneio europeu (como **Julho** e **Agosto**), além de não observar grandes tendências lineares de crescimento nos últimos anos.

<div align="center">
  <img src="images/output_65_12.png" width="80%" alt="Médias Móveis de Consumo">
</div>

---

## 🤖 Modelagem e Resultados

Para determinar que o modelo escolhido é de fato útil, estabelecemos um modelo de base (Baseline): **A média móvel simples de 7 dias**, cuja simulação resultou em um **Erro Percentual Absoluto Médio (MAPE) de ~6.02%**.

### 1. Modelos Clássicos (ARIMA e SARIMA)
A série temporal original não era estacionária ($p$-value do teste ADF > 0.05). Após aplicar a primeira diferenciação e usar os gráficos de ACF e PACF, criamos arquiteturas iterativas. O melhor modelo clássico testado, que superou a baseline e previu corretamente a sazionalidade semanal, foi um modelo modular **SARIMA (1,1,1)(0,1,2)[7]**.

### 2. Modelos de Machine Learning (O Foco!)
Devido ao vasto volume de dados histíricos, os modelos de árvore provaram ser incrivelmente performáticos. Dividimos a abordagem de features de duas formas:
- **Abordagem A:** Features extraídas por Data/Calendário (*Dia da semana, Dia do ano, etc.*).
- **Abordagem B:** Features extraídas por Lags (janelas defasadas).

Testamos **Ridge Regression, Random Forest e XGBoost**. Os algoritmos não lineares se saíram excepcionalmente bem. A *Abordagem A* (Calendário) acompanhada do modelo XGBoost foi a preferida para simular a prova de hiperparâmetros, pois era mais adaptável às dinâmicas futuras que a simples repetição do passado.

<div align="center">
  <img src="images/output_206_38.png" width="60%" alt="Feature Importance">
</div>
*Gráfico de Importância de Features (XGBoost) revelando a enorme dependência temporal baseada no "Dia do Ano" (DayofYear).*

### � Tabela de Performance

| Modelo Aplicado | MAE | MAPE (%) |
| :--- | :---: | :---: |
| Baseline (Média 7 dias) | 241,342.49 | 6.02 |
| ARIMA (Auto) | 236,505.08 | 5.93 |
| Regressão Ridge (Lags) | 103,340.00 | 2.53 |
| Random Forest (Lags) | 103,340.00 | 2.53 |
| XGBoost (Features Data) | 118,992.46 | 2.84 |

Após a etapa de busca de hiperparâmetros (Hyperparameter Tuning com Cross Validation TimeSeriesSplit), o modelo final **XGBoost Tuned** foi testado resultando numa estabilização de **MAPE em torno de 3.3%**.

> 💡 Isso representa uma **redução de erro de quase 50%** comparado a métrica de previsão do Baseline de negócio! Em um cenário de gestão energética, essa precisão reflete na enorme economia de recursos públicos.

---

## 🔮 Conclusões e Previsão Futura
Aplicando o modelo treinado a um cenário de dados não conhecidos, projetamos com sucesso o consumo dos 60 dias subsequentes. 

<div align="center">
  <img src="images/output_221_39.png" width="80%" alt="Previsão para 60 Dias">
</div>

Ao longo deste repositório, ficou provado o impacto gigantesco da exploração atenciosa dos dados para identificação de sazionalidades, bem como do poder da Modelagem de Dados Moderna sobre a previsão estática do passado.

### Próximos Passos (Extras)
- Realizar deploy das predições com integração a uma interface Flask / FastAPI.
- Rodar experimentações adicionais utilizando **Prophet (Meta)** ou Algoritmos de Deep Learning.
- Expandir a previsão para múltiplas regiões ou cruzamento de todos os modais de forma integrada.
