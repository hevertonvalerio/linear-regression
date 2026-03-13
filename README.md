# Regressão Linear Simples

Projeto simples de Regressão Linear em Python utilizando NumPy, Pandas, Matplotlib e Scikit-learn.

## 📦 Tecnologias

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 🚀 Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/hevertonvalerio/linear-regression.git
cd linear-regression
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o projeto

```bash
python main.py
```

## 📊 O que o projeto faz

- Gera dados sintéticos de preço de imóveis com base na metragem (m²)
- Treina um modelo de Regressão Linear
- Exibe as métricas do modelo (R², MAE, RMSE)
- Plota o gráfico com os dados e a reta de regressão
- Permite fazer predições com novos valores

## 📈 Exemplo de saída

```
=== Regressão Linear: Preço x Área (m²) ===
Coeficiente (inclinação): 3521.45
Intercepto: 48203.12

=== Métricas do Modelo ===
R² Score:  0.9823
MAE:       8234.56
RMSE:      10541.23

Predição para 80 m²: R$ 329.919,12
Predição para 120 m²: R$ 470.777,12
```
