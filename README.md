# Alerta temprana de cartera

**Quórum Risk · Analítica predictiva aplicada a la gestión de cartera**

> **Estado:** proyecto en preparación. Este repositorio todavía no contiene modelos entrenados, resultados de evaluación ni una aplicación funcional.

## ¿Qué buscamos resolver?

En una cartera de clientes que pagan a crédito o a plazos, el deterioro suele hacerse visible cuando el atraso ya ocurrió. El objetivo de este proyecto es explorar cómo identificar **señales tempranas de cambio en el comportamiento de pago** para ayudar a los equipos a decidir **a quién dar seguimiento, cuándo y con qué prioridad**.

El enfoque está pensado para organizaciones de distintos sectores: entidades financieras, constructoras, distribuidores, comercios y empresas de servicios con cartera.

## Alcance previsto

El desarrollo se organizará progresivamente en cuatro componentes:

1. **Comprensión de los datos:** disponibilidad, calidad, historial de pagos y definición del evento de deterioro.
2. **Análisis exploratorio:** patrones de puntualidad, segmentaciones y cambios de comportamiento.
3. **Modelado predictivo:** comparación de reglas o modelos base con técnicas de Machine Learning, cuando el volumen y la calidad de los datos lo justifiquen.
4. **Decisiones y seguimiento:** criterios de priorización, evaluación temporal, calibración, monitoreo y una posible visualización ejecutiva.

La solución concreta, el horizonte de predicción y las métricas se definirán según la necesidad y las restricciones de los datos. **No se garantiza una mejora ni se presume que Machine Learning sea siempre necesario.**

## Principios técnicos

- Separar adecuadamente entrenamiento, validación y prueba, respetando el tiempo.
- Prevenir fugas de información (*data leakage*).
- Evaluar desbalance, precisión, cobertura y calibración según el costo de las decisiones.
- Comparar contra métodos base antes de añadir complejidad.
- Documentar supuestos, limitaciones y condiciones de uso.
- Evitar decisiones automáticas sobre personas sin revisión y controles apropiados.

## Hoja de ruta

- [x] Crear el repositorio e incluir su documentación inicial.
- [ ] Definir un caso de uso demostrativo y los criterios de éxito.
- [ ] Incorporar datos **públicos o sintéticos** aptos para su publicación.
- [ ] Construir análisis exploratorio y línea base reproducible.
- [ ] Entrenar y evaluar modelos, si procede.
- [ ] Documentar resultados verificables y limitaciones.
- [ ] Preparar una demostración de decisiones o visualización.

## Confidencialidad

Este repositorio **no debe contener información de clientes, bases de datos bancarias, código propietario, credenciales ni resultados internos de empleadores o terceros**. Los ejemplos futuros deberán usar información autorizada para divulgación, pública o sintética, e identificar claramente su procedencia.

Los ejemplos y resultados que eventualmente se publiquen serán demostrativos; no deben presentarse como resultados obtenidos para clientes reales.

## Contacto

**Quórum Risk** — Inteligencia de cartera con datos y criterio.  
[contacto@quorumrisk.co](mailto:contacto@quorumrisk.co)

---

### English summary

**Early-warning analytics for payment portfolios — project in preparation.**

This repository will explore how payment-history signals can inform early detection of deterioration and prioritize customer follow-up for businesses offering credit or installment payments. Scope, evaluation metrics and prediction horizons will be defined only after validating the use case and available data. Future examples will use public or synthetic data; no client or employer-confidential information will be published.
