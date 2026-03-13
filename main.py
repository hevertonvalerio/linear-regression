import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. Geração de dados sintéticos
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 200

area = np.random.uniform(30, 200, n_samples)          # m² entre 30 e 200
noise = np.random.normal(0, 15000, n_samples)          # ruído realista
preco = 3500 * area + 50000 + noise                    # preço em R$

df = pd.DataFrame({'area_m2': area, 'preco_rs': preco})
print("=== Primeiras linhas do dataset ===")
print(df.head())
print(f"\nTotal de amostras: {len(df)}")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2. Divisão treino / teste
# ─────────────────────────────────────────────
X = df[['area_m2']]
y = df['preco_rs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 3. Treinamento do modelo
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== Regressão Linear: Preço x Área (m²) ===")
print(f"Coeficiente (inclinação): {model.coef_[0]:.2f}")
print(f"Intercepto:               {model.intercept_:.2f}")

# ─────────────────────────────────────────────
# 4. Avaliação do modelo
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Métricas do Modelo ===")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:,.2f}")
print(f"RMSE:      {rmse:,.2f}")

# ─────────────────────────────────────────────
# 5. Predições com novos valores
# ─────────────────────────────────────────────
novos = pd.DataFrame({'area_m2': [50, 80, 100, 120, 150]})
predictions = model.predict(novos)

print("\n=== Predições ===")
for area_val, pred in zip(novos['area_m2'], predictions):
    print(f"  {area_val} m²  →  R$ {pred:,.2f}")

# ─────────────────────────────────────────────
# 6. Visualização
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Regressão Linear — Preço x Área (m²)', fontsize=14, fontweight='bold')

# Gráfico 1: Dados + reta de regressão
ax1 = axes[0]
ax1.scatter(X_train, y_train, alpha=0.5, color='steelblue', label='Treino', s=20)
ax1.scatter(X_test,  y_test,  alpha=0.7, color='orange',    label='Teste',  s=20)
x_line = np.linspace(30, 200, 300).reshape(-1, 1)
y_line = model.predict(x_line)
ax1.plot(x_line, y_line, color='red', linewidth=2, label='Reta de Regressão')
ax1.set_xlabel('Área (m²)')
ax1.set_ylabel('Preço (R$)')
ax1.legend()
ax1.set_title('Dados e Reta de Regressão')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'R${v/1000:.0f}k'))

# Gráfico 2: Valores reais vs preditos
ax2 = axes[1]
ax2.scatter(y_test, y_pred, alpha=0.6, color='purple', s=20)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal')
ax2.set_xlabel('Valores Reais (R$)')
ax2.set_ylabel('Valores Preditos (R$)')
ax2.set_title(f'Real vs Predito  (R²={r2:.4f})')
ax2.legend()
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'R${v/1000:.0f}k'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'R${v/1000:.0f}k'))

plt.tight_layout()
plt.savefig('regressao_linear.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico salvo em: regressao_linear.png")
