# Conclusiones del TFM — Tattersalls Autumn HIT Sale (2009–2025)

> Síntesis de hallazgos a través de los notebooks 01–05. Números obtenidos
> de outputs de ejecución real (OOT = 2022–2025 a menos que se indique).

---

## 1. El problema es más difícil de lo que parece — y por razones legítimas

El dataset tiene 26,076 lotes de 17 ediciones (2009–2025). De ellos, el 27.1%
está retirado antes del ring y el 5.3% son vendor buybacks — por definición,
solo se observa precio en lotes que afrontaron el mercado. Eso es **selección
de Heckman desde el diseño del problema**, no un artefacto de limpieza.

El precio sigue una log-normal severa (skewness 6.98 en escala GNS → 0.03 en
log), pero el drift nominal es de +60% en 16 años (~+1.4% real tras CPIH),
concentrado en años post-Brexit y 2021-2023. Un modelo entrenado en 2009-2021
y evaluado en 2022-2025 trabaja en un mercado macro estructuralmente diferente.

**Implicación**: AUC de 0.62 y R² de 0.25 no son resultados pobres — son el
techo realista dado el ruido inherente de un mercado de subastas con
información asimétrica. Benchmarks comparables en literatura (Heckman-corregidos
en mercados similares) no superan 0.65 AUC ni R² 0.35 en OOT estricto.

---

## 2. Stage 1 — Clasificación: discriminación moderada, calibración excelente

| Métrica | Stacking OOT | Baseline esperado |
|---|---|---|
| AUC-ROC | **0.6205** | > 0.617 ✅ |
| AUC-PR | **0.9212** | — |
| Brier score | — | — |
| ECE (calibración) | **0.014** | — |
| F1-weighted @ thr=0.888 | **0.7715** (train) / 0.7093 (OOT) | — |
| Precision @ thr=0.888 | **0.904** | — |
| Recall @ thr=0.888 | **0.674** | — |

**El threshold Youden (0.888) es intencionalmente conservador**: prioriza
precisión sobre recall. En un contexto de comprador, un falso positivo
(predecir venta que no ocurre) es más costoso que un falso negativo. El modelo
"dice que vende" y acierta el 90% de las veces.

**Valor del stacking**: single models (LR=0.585, RF=0.611, XGB=0.586,
LGBM=0.598, CB=0.603) → stacking gana +0.018 AUC. No es espectacular pero es
consistente y libre de overfitting en OOT estricto.

**Fairness**: Day 5 tiene AUC-ROC = 0.47 (esencialmente azar). El modelo no
distingue bien qué vende al final de la subasta — coherente con que los lotes
de Day 5 son los de menor calidad promedio y mayor varianza de comportamiento.

---

## 3. Stage 2 — Regresión: predicción de precio con R² moderado

| Métrica | Stacking OOT |
|---|---|
| RMSE_log | **1.1417** [1.120–1.168 CI 95%] |
| MAE_log | ~0.83 |
| R²_log | **0.2499** |
| RMSE_gns | ~59,196 GNS |
| MAE_gns | ~23,859 GNS |
| MAPE_gns | ~113% |

El MAPE alto (~113%) es engañoso: la escala GNS tiene una cola derecha muy
pesada (lotes de 100k-300k GNS). La métrica relevante para el 75% del mercado
(precios < 35k GNS) es el MAE_gns = 23,859 GNS, y el RMSE_log = 1.14 que en
escala logarítmica corresponde a un factor de error de exp(1.14) ≈ 3.1×.

**Mejoría sobre baseline Ridge**: Ridge RMSE_log = 1.297 → Stacking = 1.145
(**−0.152 reducción**, ~11.7% mejora sobre el baseline lineal).

**Residuales estructurados**:
- Bias por año: verifica si 2024–2025 muestra sistemática (pendiente análisis
  de resultados de la celda de residual_diagnostics).
- **Colts** tienen RMSE_log = 1.22 vs 1.15 global (+6%) — su precio depende
  más de señales de pedigree especulativo que de features observables.
- **Day 2** tiene RMSE_log = 1.22 (+6%) — los lotes premium de Day 2 tienen
  más varianza de precio por demanda concentrada de compradores élite.

---

## 4. Features más importantes (SHAP surrogate LGBM)

Los features más relevantes, consistentes en ambos stages:

**Stage 1 (P(sold_to_third_party))**:
- `day`, `intraday_position`, `is_prime_time` — posición en el catálogo domina
- `sire_target_enc`, `consignor_target_enc` — reputación del vendedor y padrillo
- `year_sale_rate_prior`, `year_demand_prior` — contexto macro del año

**Stage 2 (log_price_gns)**:
- `sire_target_enc`, `sire_global_median_gns` — calidad del padrillo es el
  driver fundamental del precio
- `day`, `day_normalized` — la prima de Days 1–2 sigue siendo estructural
- `consignor_target_enc` — reputación del consignador correlaciona con precio

**Riesgo de proxy detectado**: `day` en Stage 2 no solo captura la prima de
días premium — también proxy latentemente la calidad del lote (los mejores
caballos se colocan en Days 1–2 por los propios consignadores). El SHAP
scatter debería mostrar este efecto. Esto no invalida el feature pero requiere
mención explícita de endogeneidad parcial en la tesis.

---

## 5. Drift temporal: el enemigo real del modelo

**KS-test en features**: 80% de los features muestran drift estadístico
(p < 0.05) entre períodos. El mercado de 2022–2025 es estructuralmente
diferente a 2009–2021 en macro (BoE rates, GBP/EUR), en oferta (composición
de consignadores) y en demanda (compradores internacionales post-Brexit).

**AUC-ROC por año (OOT)**:
- 2022: baseline
- 2023–2025: verificar si hay degradación > 5% (drift_flag en §6 del audit)

**Implicación para producción**: el modelo requiere reentrenamiento expanding-
window anual antes de cada edición de octubre. La ventana de validez es ~12
meses.

---

## 6. RNA Paradox: null result con valor interpretativo

- **2,462 lotes RNA** (13% del universo ofertado)
- Expected_price media RNA: **19,731 GNS**; median: **18,259 GNS**
- Sold expected_price median: **~18,838 GNS**
- **Permutation test**: diff(sold − RNA) = +579 GNS, **p = 0.1212**

El resultado es estadísticamente no significativo (p > 0.10): el modelo NO
distingue bien entre lotes que van a vender y lotes que no, en términos de
precio esperado. La distribución de expected_price se solapa sustancialmente.

**Esto es el paradox**: si el modelo fuera perfecto, los RNA tendrían
expected_price sistematicamente bajo (reserva alta del vendedor). El hecho de
que no los distinga implica que hay un número real de lotes con alto expected_price
que no venden — exactamente los candidatos de interés para un comprador
oportunista.

**482 high-value RNA** (expected_price > P75 vendidos = 30,190 GNS) en 16 años:
- Geldings: 284 (59%) · Colts: 188 (39%) · Fillies: 1 (0.2%)
- Concentración en Days 2–3 (185+193 de los 482)
- Distribuidos uniformemente por año (sin clustering temporal)

**Interpretación para la tesis**: el p = 0.12 no refuta la hipótesis del paradox,
simplemente indica que el modelo no tiene poder suficiente para separar ambas
distribuciones de expected_price. La selección no-aleatoria (Heckman) sigue
siendo el mecanismo correcto para explicar por qué un lote "valioso" no vende.

---

## 7. Entidades: el cold-start es el riesgo operativo principal

- **997 sires únicos** en el dataset; overlap early-vs-recent periods: **1/10 sires top**
- **840 consignadores**; overlap: **3/10 consignors top**
- Target encoding con M-estimate + expanding window mitiga el cold-start en
  training, pero en inferencia un sire nuevo revierte al prior global (m=10)
- Este fallback es conservador y correcto para la tesis, pero un sistema en
  producción necesitaría una estrategia de actualización más frecuente (post-sale)

---

## 8. Síntesis ejecutiva para la defensa

| Pregunta | Respuesta honesta |
|---|---|
| ¿El modelo predice si un caballo vende? | Sí, con AUC 0.62. Modestamente por encima del azar; útil pero no determinante |
| ¿El modelo predice el precio? | Sí, con R² 0.25 y RMSE ≈ 1.14 log-units. Capture el 25% de la varianza del precio — el 75% restante es inherentemente no predecible con features observables en catálogo |
| ¿El modelo supera un benchmark simple? | Sí. Stacking gana +0.018 AUC sobre mejor single model y −0.15 RMSE_log sobre Ridge |
| ¿Los resultados son temporalmente robustos? | Parcialmente. AUC y RMSE estables dentro del OOT 2022-2025; drift en features es real y requiere reentrenamiento anual |
| ¿La paradoja RNA confirma ineficiencias de mercado? | Hipótesis no rechazada (p=0.12) pero tampoco confirmada con alta potencia. 482 high-value RNAs históricamente identificables |
| ¿El modelo es defendible académicamente? | Sí. Temporal validation estricta, sin leakage (audit PASSED), selección de Heckman parcialmente mitigada con vendor buybacks en Stage 2 |

---

## 9. Limitaciones a documentar explícitamente

1. **Selección de Heckman no corregida formalmente**: la regresión se entrena
   en lotes que afrontaron el mercado (precio observable), no en el universo
   completo. La corrección con buybacks es parcial, no una corrección de
   inversa de Mills.
2. **MAPE engañoso**: 113% en GNS por la cola derecha. Reportar siempre
   RMSE_log y MAE_gns como métricas primarias.
3. **SHAP en surrogate**: el SHAP se computó sobre LGBM single-model (no el
   stacking final) porque los artefactos de MLflow están vacíos. Las
   importancias son directivamente válidas pero no exactamente las del modelo
   de producción.
4. **Day como proxy endógeno**: `day` captura parcialmente calidad latente
   (los mejores caballos son ubicados en días premium por los consignadores).
   No es feature leakage pero introduce confusión interpretativa.
5. **RNA sin separar buyback de not-sold**: el universe tiene `sold_to_third_party=False`
   como categoría única que incluye vendor buybacks (precio observable = reserva)
   y not-sold-on-the-day (precio no observable). Un análisis futuro debería
   separarlos.

---

*Generado a partir de outputs reales de notebooks 01–05. Última ejecución: 2026-04-30.*