import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# Configuración del modelo Monte Carlo
np.random.seed(42)  # Para reproducibilidad
n_simulations = 1000


# 1. DEFINICIÓN DE PARÁMETROS BASE Y SUS DISTRIBUCIONES

# Factores de impacto con distribuciones de probabilidad
def generate_factor_distributions():
    """
    Genera distribuciones de probabilidad para cada factor de impacto
    basadas en estimaciones de la industria farmacéutica
    """

    # A. IMPACTO REGULATORIO (distribución Beta para valores acotados 0-1)
    reg_datamat = np.random.beta(9, 1, n_simulations)  # Media ~0.9, baja varianza
    reg_dataia = np.random.beta(3, 7, n_simulations)  # Media ~0.3, alta varianza
    reg_ethics = np.random.beta(6, 4, n_simulations)  # Media ~0.6, varianza media

    # B. IMPACTO OPERACIONAL (distribución normal truncada)
    op_datamat = np.clip(np.random.normal(0.8, 0.12, n_simulations), 0, 1)
    op_dataia = np.clip(np.random.normal(0.4, 0.15, n_simulations), 0, 1)
    op_ethics = np.clip(np.random.normal(0.3, 0.09, n_simulations), 0, 1)

    # C. RIESGO FINANCIERO (distribución log-normal para capturar colas pesadas)
    risk_datamat = np.clip(np.random.lognormal(np.log(0.85), 0.2, n_simulations), 0, 1)
    risk_dataia = np.clip(np.random.lognormal(np.log(0.15), 0.3, n_simulations), 0, 1)
    risk_ethics = np.clip(np.random.lognormal(np.log(0.65), 0.25, n_simulations), 0, 1)

    # D. VENTAJA COMPETITIVA (distribución triangular)
    comp_datamat = np.random.triangular(0.3, 0.5, 0.7, n_simulations)
    comp_dataia = np.random.triangular(0.6, 0.8, 1.0, n_simulations)
    comp_ethics = np.random.triangular(0.4, 0.6, 0.8, n_simulations)

    return {
        'regulatorio': {'datamat': reg_datamat, 'dataia': reg_dataia, 'ethics': reg_ethics},
        'operacional': {'datamat': op_datamat, 'dataia': op_dataia, 'ethics': op_ethics},
        'riesgo': {'datamat': risk_datamat, 'dataia': risk_dataia, 'ethics': risk_ethics},
        'competitivo': {'datamat': comp_datamat, 'dataia': comp_dataia, 'ethics': comp_ethics}
    }


# 2. PESOS DE LOS FACTORES CON INCERTIDUMBRE

def generate_factor_weights():
    """
    Genera pesos de factores con variabilidad basada en contexto empresarial
    """
    # Distribución Dirichlet para asegurar que sumen 1
    base_weights = [0.35, 0.30, 0.20, 0.15]  # regulatorio, operacional, riesgo, competitivo
    alpha = np.array(base_weights) * 50  # Parámetro de concentración

    weights_matrix = np.random.dirichlet(alpha, n_simulations)

    return {
        'regulatorio': weights_matrix[:, 0],
        'operacional': weights_matrix[:, 1],
        'riesgo': weights_matrix[:, 2],
        'competitivo': weights_matrix[:, 3]
    }


# 3. FACTOR DE MADUREZ ORGANIZACIONAL

def generate_maturity_factors():
    """
    Factores de madurez con incertidumbre basada en evaluación organizacional (AS-IS).
    Los factores ajustan el impacto de cada área según su nivel de madurez actual
    y el potencial de desarrollo organizacional.
    """
    # Listas de datos de evaluación de madurez (escala 1-5)
    data_mat = [3.2, 3.2, 2.8, 2, 2.6, 2.4, 3, 3, 2.4, 3.6, 1.4, 2.2, 2.2, 2.4, 2.6, 2, 2.2, 2.8, 1.8, 1.6, 2.4]
    data_ia = [2, 1, 1.3, 1.6, 2, 1.3, 1, 1, 2, 1, 3.3, 1.3, 1.6, 1.6, 1, 1.6]
    data_ethics = [2.3, 1.3, 2, 1.3, 2.7, 3.7, 3.3, 1, 2]

    # Función para calcular el promedio
    def calcular_promedio(lista):
        return sum(lista) / len(lista)

    # Cálculo de promedios
    promedio_mat = calcular_promedio(data_mat)
    promedio_ia = calcular_promedio(data_ia)
    promedio_ethics = calcular_promedio(data_ethics)

    # DataMAT: Madurez media (2.43/5) con buen potencial de desarrollo
    # Factor ligeramente por debajo de 1 para reflejar necesidad de optimización
    # pero con capacidad establecida que permite crecimiento sostenido
    mat_factor = np.random.gamma(promedio_mat, 0.52, n_simulations)  # Media ~1.26

    # DataIA: Madurez muy baja (1.51/5) con gran potencial pero alta incertidumbre
    # Factor significativamente por debajo de 1 debido a:
    # - Falta de expertise técnico consolidado
    # - Infraestructura en desarrollo inicial
    # - Riesgos operacionales elevados en fase temprana
    ia_factor = np.random.gamma(promedio_ia, 0.58, n_simulations)  # Media ~0.88

    # DataEthics: Madurez baja (2.07/5) pero con base sólida para crecimiento
    # Factor moderadamente por debajo de 1, reflejando:
    # - Marcos éticos establecidos pero no completamente implementados
    # - Necesidad de fortalecimiento en procesos y cultura organizacional
    # - Potencial de mejora significativo con inversión adecuada
    ethics_factor = np.random.gamma(promedio_ethics, 0.27, n_simulations)  # Media ~0.60

    return {
        'datamat': mat_factor,
        'dataia': ia_factor,
        'ethics': ethics_factor
    }


# 4. FUNCIÓN PRINCIPAL DE SIMULACIÓN

def run_monte_carlo():
    """
    Ejecuta la simulación Monte Carlo completa
    """
    # Generar todas las distribuciones
    impacts = generate_factor_distributions()
    weights = generate_factor_weights()
    maturity = generate_maturity_factors()

    # Arrays para almacenar resultados
    scores_datamat = np.zeros(n_simulations)
    scores_dataia = np.zeros(n_simulations)
    scores_ethics = np.zeros(n_simulations)

    # Simulación iterativa
    for i in range(n_simulations):
        # Cálculo de scores ponderados para cada simulación
        score_datamat = (
                impacts['regulatorio']['datamat'][i] * weights['regulatorio'][i] +
                impacts['operacional']['datamat'][i] * weights['operacional'][i] +
                impacts['riesgo']['datamat'][i] * weights['riesgo'][i] +
                impacts['competitivo']['datamat'][i] * weights['competitivo'][i]
        )

        score_dataia = (
                impacts['regulatorio']['dataia'][i] * weights['regulatorio'][i] +
                impacts['operacional']['dataia'][i] * weights['operacional'][i] +
                impacts['riesgo']['dataia'][i] * weights['riesgo'][i] +
                impacts['competitivo']['dataia'][i] * weights['competitivo'][i]
        )

        score_ethics = (
                impacts['regulatorio']['ethics'][i] * weights['regulatorio'][i] +
                impacts['operacional']['ethics'][i] * weights['operacional'][i] +
                impacts['riesgo']['ethics'][i] * weights['riesgo'][i] +
                impacts['competitivo']['ethics'][i] * weights['competitivo'][i]
        )

        # Aplicar factores de madurez
        score_datamat *= maturity['datamat'][i]
        score_dataia *= maturity['dataia'][i]
        score_ethics *= maturity['ethics'][i]

        # Almacenar scores
        scores_datamat[i] = score_datamat
        scores_dataia[i] = score_dataia
        scores_ethics[i] = score_ethics

    return scores_datamat, scores_dataia, scores_ethics, impacts, weights, maturity


# 5. NORMALIZACIÓN Y CÁLCULO DE PONDERACIONES

def calculate_weights(scores_datamat, scores_dataia, scores_ethics):
    """
    Calcula las ponderaciones normalizadas para cada simulación
    """
    # Normalización por simulación
    weights_datamat = np.zeros(n_simulations)
    weights_dataia = np.zeros(n_simulations)
    weights_ethics = np.zeros(n_simulations)

    for i in range(n_simulations):
        total = scores_datamat[i] + scores_dataia[i] + scores_ethics[i]
        weights_datamat[i] = scores_datamat[i] / total
        weights_dataia[i] = scores_dataia[i] / total
        weights_ethics[i] = scores_ethics[i] / total

    return weights_datamat, weights_dataia, weights_ethics


# 6. ANÁLISIS ESTADÍSTICO

def analyze_results(weights_datamat, weights_dataia, weights_ethics):
    """
    Análisis estadístico completo de los resultados
    """
    results = {
        'DataMAT': {
            'mean': np.mean(weights_datamat),
            'std': np.std(weights_datamat),
            'median': np.median(weights_datamat),
            'ci_95': np.percentile(weights_datamat, [2.5, 97.5]),
            'min': np.min(weights_datamat),
            'max': np.max(weights_datamat)
        },
        'DataIA': {
            'mean': np.mean(weights_dataia),
            'std': np.std(weights_dataia),
            'median': np.median(weights_dataia),
            'ci_95': np.percentile(weights_dataia, [2.5, 97.5]),
            'min': np.min(weights_dataia),
            'max': np.max(weights_dataia)
        },
        'DataEthics': {
            'mean': np.mean(weights_ethics),
            'std': np.std(weights_ethics),
            'median': np.median(weights_ethics),
            'ci_95': np.percentile(weights_ethics, [2.5, 97.5]),
            'min': np.min(weights_ethics),
            'max': np.max(weights_ethics)
        }
    }

    return results


# 7. VISUALIZACIÓN

def create_visualizations(weights_datamat, weights_dataia, weights_ethics):
    """
    Crea visualizaciones de los resultados Monte Carlo
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Histogramas de distribuciones
    ax1.hist(weights_datamat * 100, bins=50, alpha=0.7, label='DataMAT', color='blue')
    ax1.hist(weights_dataia * 100, bins=50, alpha=0.7, label='DataIA', color='red')
    ax1.hist(weights_ethics * 100, bins=50, alpha=0.7, label='DataEthics', color='green')
    ax1.set_xlabel('Ponderación (%)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Ponderaciones - Monte Carlo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plots
    data_box = [weights_datamat * 100, weights_dataia * 100, weights_ethics * 100]
    ax2.boxplot(data_box, labels=['DataMAT', 'DataIA', 'DataEthics'])
    ax2.set_ylabel('Ponderación (%)')
    ax2.set_title('Box Plot de Ponderaciones')
    ax2.grid(True, alpha=0.3)

    # Scatter plot de correlaciones
    ax3.scatter(weights_datamat * 100, weights_dataia * 100, alpha=0.5, s=10)
    ax3.set_xlabel('DataMAT (%)')
    ax3.set_ylabel('DataIA (%)')
    ax3.set_title('Correlación DataMAT vs DataIA')
    ax3.grid(True, alpha=0.3)

    # Series temporales (simulación a simulación)
    ax4.plot(weights_datamat * 100, alpha=0.7, label='DataMAT', linewidth=0.5)
    ax4.plot(weights_dataia * 100, alpha=0.7, label='DataIA', linewidth=0.5)
    ax4.plot(weights_ethics * 100, alpha=0.7, label='DataEthics', linewidth=0.5)
    ax4.set_xlabel('Simulación')
    ax4.set_ylabel('Ponderación (%)')
    ax4.set_title('Evolución por Simulación')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 8. EJECUCIÓN COMPLETA

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("Ejecutando simulación Monte Carlo...")
    print(f"Número de simulaciones: {n_simulations}")
    print("-" * 50)

    # Ejecutar simulación
    scores_datamat, scores_dataia, scores_ethics, impacts, weights, maturity = run_monte_carlo()

    # Calcular ponderaciones
    weights_datamat, weights_dataia, weights_ethics = calculate_weights(
        scores_datamat, scores_dataia, scores_ethics
    )

    # Análisis estadístico
    results = analyze_results(weights_datamat, weights_dataia, weights_ethics)

    # Mostrar resultados
    print("RESULTADOS MONTE CARLO:")
    print("=" * 50)

    for component, stats in results.items():
        print(f"\n{component}:")
        print(f"  Media: {stats['mean']:.3f} ({stats['mean'] * 100:.1f}%)")
        print(f"  Mediana: {stats['median']:.3f} ({stats['median'] * 100:.1f}%)")
        print(f"  Desv. Estándar: {stats['std']:.3f}")
        print(f"  IC 95%: [{stats['ci_95'][0]:.3f}, {stats['ci_95'][1]:.3f}]")
        print(f"  Rango: [{stats['min']:.3f}, {stats['max']:.3f}]")

    # Validación de suma = 1
    sum_check = np.mean(weights_datamat + weights_dataia + weights_ethics)
    print(f"\nValidación (suma = 1): {sum_check:.6f}")

    # Crear visualizaciones
    create_visualizations(weights_datamat, weights_dataia, weights_ethics)

    return results, weights_datamat, weights_dataia, weights_ethics


# Ejecutar el análisis
if __name__ == "__main__":
    results, w_mat, w_ia, w_eth = main()

    # Comparación con ponderación propuesta
    print("\n" + "=" * 50)
    print("COMPARACIÓN CON PONDERACIÓN PROPUESTA (60-20-20):")
    print("=" * 50)

    proposed = [0.60, 0.20, 0.20]
    mc_means = [results['DataMAT']['mean'], results['DataIA']['mean'], results['DataEthics']['mean']]

    for i, (component, prop, mc) in enumerate(zip(['DataMAT', 'DataIA', 'DataEthics'], proposed, mc_means)):
        diff = abs(prop - mc)
        print(f"{component}: Propuesto {prop:.1%} vs MC {mc:.1%} (Diferencia: {diff:.1%})")

    # Error cuadrático medio
    mse = np.mean([(p - m) ** 2 for p, m in zip(proposed, mc_means)])
    print(f"\nError Cuadrático Medio: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.3f} ({np.sqrt(mse) * 100:.1f}%)")
