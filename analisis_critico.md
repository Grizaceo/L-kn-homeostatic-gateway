# Auditoría Crítica: L-kn Gateway + Geometría Tropical + SGLang/RadixAttention (Brief v0.3)

**Fecha de análisis:** 2026-02-05  
**Método:** Evaluación según marco de rigor §1–§6 del protocolo de investigación  
**Documentos base:** Brief v0.3, Resumen Ejecutivo, Estado del Arte Tropical en ML, TropiUMAP/Isomap MVP, Arquitectura MöbiusRAS-Sinthome, Entregable A, Agente Local, papers adjuntos

---

## 1. Diagnóstico general: contradicción interna crítica

Antes de entrar al análisis por componente, hay un problema estructural que atraviesa todo el proyecto y que debe resolverse antes de experimentar.

**El Brief v0.3 y el Resumen Ejecutivo se contradicen directamente.** El Resumen Ejecutivo (que parece ser una evaluación posterior y más madura) dictaminó:

- Pruning tropical: **NO-GO** por ahora
- Políticas tropicales de KV eviction: **NO-GO** inicial
- Librería `tropica`: **POSPONER**
- Control L-kn v2 (aprendido): **Posponer**

Sin embargo, el Brief v0.3 construye *todo* el sistema sobre geometría tropical como pieza central (probe tropical, signed-tropical, lifting/dequantization, honeycombs como gramática de factibilidad). No hay reconciliación explícita entre ambos documentos.

**Veredicto:** O bien el Brief v0.3 representa una evolución *posterior* al Resumen Ejecutivo que justifica reintroducir lo tropical con un enfoque diferente (como herramienta de *gating* ligero, no de KV eviction), o bien hay un drift conceptual no controlado. Esta ambigüedad debe resolverse explícitamente antes de invertir tiempo en implementación.

---

## 2. Análisis componente por componente

### 2.1 Probe tropical para FLUIDO vs ANALÍTICO (Brief §1.1–1.2)

**2.1.1 Qué afirma realmente**

Promete: un score piecewise-linear barato `s(x) = max_i(b_i + a_i·x)` que, mediante gap top-1/top-2, densidad de empates (ties) y margen firmado, decide si una query necesita procesamiento ligero (FLUIDO) o pesado (ANALÍTICO).

No promete: detectar errores "confiados" (alta confianza pero incorrectos), ni sustituir verificación semántica real.

Deja de aplicar: cuando la distribución de logits/features no tiene relación monotónica con la dificultad real de la tarea, o cuando el modelo genera errores calibrados (baja entropía, alta confianza, respuesta falsa).

**2.1.2 Estado del arte**

La idea de usar entropía de logits, gap top-k, o señales de confianza como proxy de dificultad está bien establecida (A): se usa en speculative decoding, early exit, adaptive computation. Lo que se describe como "probe tropical" es, operacionalmente, un clasificador piecewise-linear sobre features de incertidumbre. Llamarlo "tropical" añade formalismo pero no añade capacidad computacional: `max(b_i + a_i·x)` es literalmente una red ReLU de una capa sin bias de salida, o equivalentemente, la función de valor de un problema de programación lineal paramétrico.

Papers relevantes que sí resuelven el problema de routing adaptativo: Raposo et al. (2024) "Mixture-of-Depths", Schuster et al. (2022) "Confident Adaptive Language Modeling", Elbayad et al. (2020) "Depth-Adaptive Transformer". Estos usan señales internas del modelo (no externas), lo que da mejor calibración.

**Lo que NO resuelven:** ninguno usa un probe *externo* piecewise-linear sobre señales agregadas. Eso es la novedad potencial, pero también el riesgo.

**2.1.3 Mecanismo operativo**

- **Qué cambia:** se introduce una evaluación O(k·d) antes de cada llamada al modelo (k = número de afines, d = dimensión de features). Si k y d son pequeños (k<10, d<50), el costo es despreciable.
- **Dónde vive:** CPU o GPU, antes del dispatch al modelo. Es un forward pass de un perceptrón lineal con activación max.
- **Costo real:** el costo no es el probe en sí, sino **obtener las features de entrada**. Si necesitas logits parciales, embeddings, o logprobs del modelo para alimentar el probe, ya pagaste buena parte del costo de inferencia. Este es el agujero lógico principal.

**2.1.4 Traslado al sistema**

La parte que **sí** puede beneficiarse: routing entre modelos de distinto tamaño (ej. 3B vs 7B), donde el probe usa features baratas (longitud del prompt, presencia de keywords, historial de la sesión) para decidir qué modelo invocar. Esto es routing a nivel de *request*, no a nivel de *token*.

La parte que **no** tiene beneficio real: routing a nivel de token basado en logprobs intermedios. En ese caso, ya estás dentro del modelo y el costo marginal de "completar la generación" es menor que el costo de haber hecho el prefill. RadixAttention no te ayuda aquí porque el prefill ya se computó.

**2.1.5 Riesgos y límites**

- **Fallo silencioso principal:** errores calibrados. Un modelo 3B puede generar respuestas incorrectas con alta confianza y bajo tie-count. El probe no dispararía escalado, y el error pasaría.
- **Empeora el sistema cuando:** los umbrales están mal calibrados. Un probe demasiado sensible escala todo (pierde la ventaja de ahorro). Uno demasiado laxo deja pasar errores.
- **Supuesto no garantizado:** que las señales de incertidumbre superficiales (gap, ties) correlacionen con la dificultad real de la tarea. Esto es (B), no (A). La correlación depende del modelo, la tarea, y la distribución de queries.

**2.1.6 Experimento mínimo**

**Hipótesis:** Un clasificador lineal (piecewise-linear con k=5 afines) sobre [prompt_length, keyword_density, session_history_length, estimated_complexity_score] logra AUC ≥ 0.75 para predecir si un modelo 3B fallará en una tarea de QA, con overhead < 1ms por request.

**Setup:**
- Modelo: Qwen 2.5 3B cuantizado, corriendo en SGLang sobre RTX 4060
- Dataset: 500 queries de QA con ground truth (ej. subconjunto de TriviaQA o Natural Questions)
- Baseline: routing aleatorio 50/50 y routing "siempre ANALÍTICO"
- Procedimiento: (1) generar respuestas con el modelo 3B, (2) evaluar correctitud, (3) loguear features del probe, (4) entrenar probe post-hoc, (5) evaluar AUC

**Métricas:** AUC del probe, latencia p50/p95 del probe, ratio de escalado (% queries enviadas a ANALÍTICO), accuracy final del pipeline vs baseline.

**Éxito:** AUC ≥ 0.75 Y ratio de escalado < 40% Y accuracy del pipeline ≥ accuracy de "siempre ANALÍTICO" - 2%.

**Refutación:** AUC < 0.65 indicaría que las features superficiales no predicen fallo, y habría que buscar features más costosas (logprobs, embeddings intermedios), lo cual puede eliminar la ventaja de latencia.

---

### 2.2 Signed-tropical como constraints (Brief §1.2)

**2.2.1 Qué afirma realmente**

Propone separar un score en componente positivo `s+(x)` y negativo `s-(x)`, ambos max-de-afines, y usar el margen `m(x) = s+(x) - s-(x)` como señal de tensión pro/contra.

**2.2.2 Estado del arte**

La idea de separar evidencia a favor y en contra es estándar en sistemas de decisión (Dempster-Shafer, belief functions, pro/con scoring). La formalización "signed tropical" tiene raíz en trabajos de Viro sobre patchworking y en extensiones del semianillo tropical, pero su uso operacional en ML es prácticamente inexistente (C).

**2.2.3 Mecanismo operativo**

Offline: entrenar/calibrar los pesos (a_k, b_k) para s+ y s- por separado. Runtime: evaluar dos max sobre vectores, restar, comparar contra umbral. Costo: 2× el probe simple, sigue siendo O(k·d).

**2.2.4 Traslado al sistema**

Esto solo tiene sentido si existen features naturalmente "pro" y "contra" en el dominio. En un gateway de routing LLM, no es obvio qué constituiría "evidencia a favor de que el modelo acertará" vs "evidencia en contra". Sin una separación semántica clara de las features, s+ y s- son simplemente dos mitades arbitrarias del mismo clasificador lineal, y el margen m(x) no aporta información nueva sobre un score único.

**2.2.5 Riesgos**

- Sobreingeniería sin ganancia: si s+ y s- se entrenan sobre las mismas features, el margen m no tiene ventaja teórica sobre un score escalar.
- La analogía con "excitación/inhibición" neuronal es (C) sin mecanismo operacional claro en este contexto.

**Veredicto: ⏸️ Posponer.** Solo vale la pena si aparece un dominio con separación natural pro/contra (ej. verificación de claims con evidence retrieval). Para el gateway de routing, un probe escalar es suficiente como primer paso.

---

### 2.3 Lifting / dequantization: log-sum-exp con temperatura (Brief §1.3)

**2.3.1 Qué afirma**

Entrenar los parámetros del probe con log-sum-exp suave (diferenciable) y en inferencia usar max duro (no diferenciable pero rápido).

**2.3.2 Estado del arte**

Esto es la "Maslov dequantization" (A), bien establecida en matemáticas tropicales. En ML, es análogo a temperature annealing en softmax, Gumbel-Softmax para discrete sampling, y straight-through estimators. Es una técnica estándar de entrenamiento, no una novedad.

**2.3.3 Mecanismo operativo**

- Entrenamiento: reemplazar max por θ·log(Σexp(·/θ)), entrenar con SGD estándar sobre (a_i, b_i).
- Inferencia: evaluar max directamente, sin exponenciales.
- Beneficio claro: gradientes no-nulos para todos los parámetros durante entrenamiento; inferencia exacta y rápida.

**2.3.4 Traslado al sistema**

Esto es directamente aplicable y operacional. Es la forma correcta de entrenar un probe piecewise-linear. No requiere ninguna infraestructura tropical especial: es un MLP de una capa con softmax temperatura→0.

**Veredicto: ✅ Usar.** Pero no presentarlo como "geometría tropical" sino como lo que es: annealing de temperatura en un clasificador lineal, técnica estándar.

---

### 2.4 Conexiones "tipo Tao": JL, compressed sensing, honeycombs (Brief §2)

**2.4.1 Johnson-Lindenstrauss para compresión**

JL (A) dice que proyecciones aleatorias de R^d a R^k preservan distancias con k = O(ε⁻²·log n). La hipótesis (B) es que el ranking coarse y ties/gap se preservan en el espacio comprimido.

**Problema concreto:** ¿qué estás comprimiendo? Si son embeddings de tokens (d=4096 para un 7B), proyectar a k=128 podría preservar distancias para n=10^4 puntos con ε=0.3. Pero las features del probe que describiste en §1.1 son de dimensión baja (d<10): no hay nada que comprimir. JL es relevante solo si el probe opera sobre embeddings de alta dimensión, lo cual contradice el principio de "probe barato".

**Veredicto: ⏸️ Investigación futura.** Solo aplica si escalas a TropiUMAP/grafos grandes. Para el gateway, es irrelevante.

**2.4.2 Compressed sensing como principio de diseño**

La analogía "encoder universal barato + decoder caro" es conceptualmente correcta (B) y describe exactamente el patrón FLUIDO→ANALÍTICO. Pero es una analogía de diseño, no un mecanismo operacional. No necesitas formalizarlo con compressed sensing para implementar un probe + escalado.

**Veredicto: ❌ Como mecanismo operacional, descartar. ✅ Como principio de diseño, mantener como guía conceptual.**

**2.4.3 Honeycombs de Knutson-Tao como "gramática de factibilidad"**

Los honeycombs (A) resuelven el problema de Horn sobre eigenvalues de sumas de matrices hermitianas. La propuesta (B) de usarlos como "inequidades compactas de consistencia entre módulos" es una analogía sin mecanismo operacional definido.

**Problema:** No se especifica qué matrices corresponden a qué módulos, qué significan los eigenvalues en el contexto del gateway, ni cómo se construyen los honeycombs a partir de telemetría del sistema. Sin esas definiciones, es especulación (C).

**Veredicto: ❌ Descartar del MVP. Posible tema teórico de largo plazo, pero requiere una formulación concreta antes de ser siquiera hipótesis.**

---

### 2.5 Error compounding multi-agente (Brief §3)

**2.5.1 Qué afirma**

Cita que sistemas multi-agente independientes amplifican errores hasta 17.2× y centralizados con orquestador hasta 4.4×. Propone checkpoints + verificación barata como mitigación.

**2.5.2 Estado del arte**

El resultado citado de Google Research (2026) es relevante (A si publicado, B si preprint). La mitigación por checkpoints y orquestador es la solución estándar. No hay nada tropical aquí: es engineering de sistemas distribuidos con validación.

**2.5.3 Agujero lógico principal**

El Brief dice que "errores confiados pueden no disparar ties/gaps" pero no propone ninguna mitigación para este caso. Si el probe tropical no detecta errores calibrados, y el sistema multi-agente amplifica errores, entonces el componente tropical del verificador es exactamente inútil en el caso donde más importa.

Las mitigaciones reales para errores confiados son: (a) verificación por herramientas externas (calculadoras, búsqueda web, tests de código), (b) consistencia entre múltiples generaciones independientes (self-consistency), (c) NLI/entailment checks. Ninguna de estas es tropical.

**2.5.4 Experimento**

**Hipótesis:** Un pipeline secuencial de 3 agentes (decompose → execute → synthesize) con checkpoints por entropía de logits reduce error amplification rate de >10× a <3× en una tarea compuesta de QA multi-hop.

**Setup:**
- Modelo: Qwen 2.5 3B, SGLang
- Dataset: 100 preguntas multi-hop (HotpotQA subset con ground truth)
- Baseline: pipeline sin checkpoints
- Tratamiento: pipeline con checkpoint que mide entropía de logits tras cada agente y re-genera si entropía > umbral

**Métricas:** accuracy final, error amplification rate (errores en paso N / errores en paso 1), latencia total, ratio de re-generaciones.

**Éxito:** accuracy mejora ≥ 10% absoluto sobre baseline sin checkpoints con ≤ 50% overhead de latencia.
**Refutación:** si el checkpoint por entropía no mejora accuracy (porque los errores son confiados), confirma que se necesitan verificadores semánticos (NLI), no estadísticos.

---

### 2.6 SGLang + RadixAttention (Brief §4)

**2.6.1 Qué afirma**

SGLang reusa KV cache por prefijos exactos (A). El pipeline FLUIDO→ANALÍTICO produce llamadas con prefijos compartidos, lo que beneficiaría de RadixAttention (B).

**2.6.2 Traslado al sistema: esto es el componente más sólido**

El match es real *si y solo si* los templates del probe y del modo analítico comparten un prefijo token-idéntico largo. Esto es factible con diseño cuidadoso de prompts (system prompt compartido + sufijos variables). El Resumen Ejecutivo ya lo marcó como GO.

**2.6.3 Agujero técnico pendiente**

No se ha medido el hit-rate real de RadixAttention en el patrón específico del gateway. El Brief §10.2 lo lista como pendiente, y sigue pendiente. Sin esta medición, la promesa de "baja TTFT" es (B).

**Verificación técnica inmediata:** un script que genere 100 requests con el patrón probe→analítico (compartiendo system prompt de 500+ tokens), mida TTFT p50/p95 con y sin RadixAttention (comparando `--disable-radix-cache` en SGLang), y reporte hit-rate del cache.

**Veredicto: ✅ GO para prototipar. Pero medir antes de asumir beneficio.**

---

### 2.7 KV cache budget en 8GB (Brief §5)

**2.7.1 Mecanismo**

La fórmula KV bytes/token = 2 · n_layers · n_kv_heads · d_head · bytes_elem es correcta (A). Para Qwen 2.5 3B (36 layers, 2 kv_heads, d_head=128, bf16): KV/token ≈ 2 · 36 · 2 · 128 · 2 = 36,864 bytes ≈ 36 KB/token. Con 4GB para KV (50% de 8GB), cabrían ~110K tokens en pool. Esto parece suficiente para el MVP.

**2.7.2 Riesgo real**

El riesgo no es el KV pool sino la concurrencia. Con 1 request a la vez (lo realista en RTX 4060), el cuello es la longitud del contexto, no el pool. Con prompts de ~2K tokens, el KV por request es ~72MB: manejable. El problema surge con branching (N>1 generaciones paralelas), que el Brief §5.2 menciona pero no cuantifica.

**Verificación inmediata:** medir VRAM real con `nvidia-smi` corriendo Qwen 2.5 3B cuantizado en SGLang con un prompt de 2K tokens. Comparar VRAM total consumido con la predicción teórica.

---

### 2.8 Topología: Möbius, toro, complejos simpliciales (Brief §8)

**2.8.1 Qué afirma**

Propone usar meshes triangulares de toro/Möbius/Klein como sustrato para routing o memoria, con métricas discretas y WL-partitions para simetría.

**2.8.2 Estado real**

Esta es la parte donde el proyecto tiene más drift conceptual. Los documentos del proyecto (Arquitectura MöbiusRAS-Sinthome, Agente Local) desarrollan extensamente la parametrización de Möbius, los pliegues extrínsecos, el gating RAS de edges, y las analogías RSI-Lacanianas. Sin embargo:

- No hay ningún resultado experimental que muestre que la topología Möbius aporta algo medible sobre un grafo plano estándar.
- La conexión Lacan→ML se etiqueta consistentemente como (C) en los propios documentos del proyecto.
- La parametrización Möbius (u,v con twist) genera un grafo que es, computacionalmente, un grafo 2D con rewiring periódico. Esto es equivalente a un small-world graph con atajos determinísticos. No necesitas la parametrización Möbius para lograr esto.

**2.8.3 Riesgos**

- **Sobreingeniería extrema:** implementar distancias geodésicas en mallas Möbius, flujos de Ricci sobre grafos, transporte paralelo discreto... para un gateway que en esencia necesita decidir "¿esta query es fácil o difícil?". La desproporción entre complejidad y función es alta.
- **La topología no es el cuello de botella:** en un sistema de inferencia LLM local, el cuello es VRAM, latencia de generación, y calidad del modelo. La topología de la memoria no es lo que limita el rendimiento.

**Veredicto: ❌ Para el MVP del gateway, descartar toda la infraestructura topológica (Möbius, toro, Klein, complejos simpliciales, RSI, Sinthome). ⏸️ Como investigación teórica independiente sobre representación de memoria en GNNs, puede tener valor, pero es un proyecto separado.**

---

### 2.9 TropiUMAP/Isomap y atlas de clusters (documento separado)

**2.9.1 Qué afirma**

Sustituir la métrica euclidiana en UMAP/Isomap por la métrica tropical d_tr, usar atlas de clusters para escalabilidad, y multi-vista para robustez.

**2.9.2 Viabilidad**

Este es un proyecto de investigación legítimo en sí mismo, con hipótesis falsables bien definidas (trustworthiness/continuity como métricas, ablations claras). Sin embargo:

- Es **independiente** del gateway L-kn. No comparte stack, métricas, ni pipeline.
- Requiere datasets de alta dimensionalidad y evaluación cuidadosa de vecindarios.
- El MVP de TropiUMAP está bien planificado en su propio documento.

**Veredicto: ⏸️ Proyecto paralelo legítimo pero que debe ejecutarse como stream separado, no como parte del gateway.**

---

## 3. Mapa de dependencias y priorización

### Lo que vale la pena prototipar ahora (✅)

1. **SGLang + RadixAttention como motor** — ya respaldado (A), solo falta verificar hit-rate empírico con el patrón de prompts del gateway.

2. **Gateway FastAPI con routing simple** — probe basado en heurísticas simples (longitud, keywords, historial), sin matemática tropical. Clasificador lineal o reglas if/then como v0. Telemetría JSON como la propuesta en §11.

3. **Entropía de logits como señal de incertidumbre** — barata (viene gratis con la generación), calibrar umbrales empíricamente.

4. **Lifting/annealing para entrenar el probe** — técnica estándar, usar sin branding "tropical".

### Lo que queda como investigación futura (⏸️)

5. **Probe tropical con features internas del modelo** — requiere acceso a logprobs/embeddings intermedios, lo cual puede eliminar la ventaja de latencia. Evaluar solo después de que el probe simple (punto 2) esté corriendo.

6. **Signed-tropical (s+/s-)** — solo si aparece un dominio con separación natural pro/contra.

7. **TropiUMAP/Isomap** — proyecto separado, bien definido, ejecutar en paralelo.

8. **JL para compresión de features** — solo relevante si el probe escala a embeddings de alta dimensión.

9. **Entropía semántica** — el Resumen Ejecutivo ya identificó el riesgo de costo. Probar con M=3-5 solo después de que entropía de logits muestre limitaciones.

### Lo que debe descartarse (❌)

10. **Honeycombs como gramática de factibilidad** — especulación sin mecanismo concreto.

11. **Topología Möbius/Sinthome/RSI para el gateway** — desproporción complejidad/beneficio, analogías (C) sin operacionalización.

12. **Políticas tropicales de KV eviction** — el Resumen Ejecutivo ya lo descartó, KeyDiff/diversidad son superiores y probados.

13. **Librería `tropica` como prerrequisito** — no construir infraestructura sin caso de uso validado.

14. **Dimensión 24 como "espacio intermedio"** — sin justificación experimental, número mágico.

---

## 4. Agujeros lógicos fundamentales (resumen)

### Agujero 1: ¿De dónde vienen las features del probe?
El probe es "barato" solo si sus inputs son baratos. Si necesitas logprobs del modelo para calcular ties/gap, ya pagaste el costo del prefill. Definir exactamente qué features son computablemente baratas *antes* de invocar al modelo.

### Agujero 2: Errores confiados no tienen mitigación
El Brief reconoce que errores "confiados" (alta confianza, respuesta incorrecta) escapan al probe tropical. Esta es precisamente la clase de errores más peligrosa en multi-agente. La mitigación real (herramientas externas, NLI, self-consistency) no es tropical y no está operacionalizada en el plan.

### Agujero 3: No hay dataset de evaluación definido
Los 5 ablations del Brief §9 no especifican qué dataset concreto se usará, ni cómo se obtiene ground truth. Sin dataset y ground truth, no hay experimento.

### Agujero 4: El branding "tropical" oscurece la operación real
Muchas de las operaciones propuestas son estándar bajo otro nombre: el probe es un clasificador lineal, el lifting es temperature annealing, las "constraints offline" son reglas de negocio, el "margen firmado" es una función de decisión bipolar. El formalismo tropical no aporta capacidad computacional adicional en estos casos. Esto no es un defecto técnico, pero sí dificulta la comunicación y la evaluación honesta del valor añadido.

### Agujero 5: Dos proyectos mezclados como uno
El gateway L-kn (routing + inferencia + telemetría) y la investigación en geometría tropical para ML (TropiUMAP, métricas tropicales, redes tropicales) son proyectos con timelines, métricas y riesgos diferentes. Mezclarlos causa que ninguno avance con foco.

---

## 5. Plan de acción recomendado (semanas 1–8)

### Semanas 1–2: Baseline funcional
- Montar SGLang + Qwen 2.5 3B cuantizado en RTX 4060
- Medir VRAM, TTFT, throughput, hit-rate de RadixAttention
- Gateway FastAPI mínimo con routing por reglas simples (longitud > umbral → ANALÍTICO)
- Telemetría JSON básica

### Semanas 3–4: Probe v0 + dataset
- Definir dataset de evaluación (100-500 queries con ground truth)
- Implementar probe como clasificador lineal sobre features baratas
- Calibrar umbrales con cross-validation
- Medir AUC, latencia, ratio de escalado

### Semanas 5–6: Checkpoints multi-paso
- Pipeline de 2-3 pasos con checkpoint por entropía de logits
- Comparar vs pipeline sin checkpoints
- Medir error amplification rate

### Semanas 7–8: Evaluación y decisión
- Compilar resultados de todos los experimentos
- Decidir si el probe justifica su overhead vs "siempre ANALÍTICO"
- Decidir si la señal de entropía de logits es suficiente o se necesita escalar a entropía semántica
- Publicar resultados y redefinir plan para fase 2

### En paralelo (si hay ancho de banda): TropiUMAP MVP
- Implementar d_tr en NumPy/Numba
- Comparar UMAP euclidiano vs tropical en 2-3 datasets estándar
- Medir trustworthiness/continuity
- Esto informa si la geometría tropical tiene valor operacional en *algún* contexto antes de integrarla al gateway

---

## 6. Conclusión

El proyecto tiene una base de ingeniería sólida (SGLang, RadixAttention, cuantización, gateway FastAPI) y una visión teórica ambiciosa. El problema principal no es falta de ideas sino exceso: hay demasiadas ideas de nivel (B) y (C) compitiendo por atención sin que ninguna haya sido validada empíricamente. La geometría tropical puede tener valor real en contextos específicos (métricas para UMAP, gating eficiente), pero el Brief v0.3 la posiciona como solución universal para routing, verificación, KV management, topología de memoria, y multi-agente, lo cual no está justificado.

La recomendación es: construir el gateway con herramientas probadas (reglas simples, entropía de logits, RadixAttention), medir todo, y *después* evaluar si algún componente tropical mejora métricas concretas sobre ese baseline. No al revés.
