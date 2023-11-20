# Modelos de Series Temporais - Análise, Modelagem Classica e Machine Learning

## 📌 Overview

O projeto em questão visa realizar o estudo de series temporais usando técnicas de mineração de dados. Esse é um assunto de extrema relevância, pois empresas, industrias, governos e demais áreas precisam frequêntemente estimar valores futuros para suas variáveis como valor de faturamento, temperatura dos próximos dias, receita de impostos e muito mais.

## 💼 Business Understanding

Como dito anteriormente, a previsão de demanda é algo extremamente importante nos mais diversos ramos de atuação. O Objeto de estudo desse projeto é relativo justamente a uma série temporal: O Consumo de energia diário da área industrial da cidade de Copenhagen, na Dinamarca.

**Serão avaliados fatores como:**

- Coleta de Dados via API
- Análise gráfica
- Reamostragem de granularidade da série
- Janelas deslizantes
- Análise estatística
- Análise de sazionalidade
- Avaliação de estacionariedade
- Decomposição de séries temporais
- Modelos AR, MA, ARIMA E SARIMA
- Forecast de series temporais
- Métricas de previsão para séries temporais (MAE e MAPE)
- Forecast via modelos machine learning
- Engenharia de dados para serie temporal

**As métricas avaliadas serão**
- MAE
- MAPE

## 📊 Modelagem

Como dito usaremos 3 principais abordagens: A primeira será o uso de modelos estatísticos classicos como ARMA, ARIMA e SARIMA. Faremos a análise gráfica, cálculo de métricas estatísticas de estacionariedade, modelagem de paramentros p,q,d,P,Q,D,S e forecast de demanda.

Passamos então para a previsão via modelos clássicos de machine learning. Para isso vamos usar as técnicas de Regressão Linear de Ridge, Random Forest e XGBostRegressor. Faremos uso de engenharia de features para criar novas variaveis que consigam representar bem os dados e com elas realizar a previsão da demanda de consumo de energia. Suas métricas de sucesso serão comparadas a outros modelos.

## 🚀 Reflexões Finais

Testamos diferentes modelagens ao decorrer do projeto. A modelagem escolhida como a vencedora para a realização da previsão foi a do XGBoost com utilização de features baseadas no dia.

Com ela conseguimos reduzir em quase **50%** o valor de MAPE do baseline calculado.

Uma redução nesse nível tem impactos significativos na previsão de energia e pode ajudar o país/empresa a ser muito mais estratégico na gestão de recursos!



