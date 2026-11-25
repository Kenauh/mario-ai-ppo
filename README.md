# Mario AI Challenge: agente PPO para el Mundo 1-1 de Super Mario Bros.

Proyecto Final 3 de Inteligencia Artificial. Un agente entrenado con **PPO** (Stable-Baselines3) que juega el Mundo 1-1 viendo solo píxeles. El agente final se ejecuta **sin SB3, Gymnasium ni NES-Py**: solo con PyTorch, NumPy y OpenCV.

- **Estudiante:** Kenauh (trabajo individual)
- **Estado:** trabajo experimental terminado; pendiente el anexo técnico del docente (interfaz del evaluador, límite de fotogramas, formato de entrega). Los parámetros del evaluador que se usan aquí son **supuestos provisionales** (ver [Limitaciones](#limitaciones-y-pendientes)).

## Resultados principales

Todas las cifras salen de ejecuciones reales; los CSV están en `results/`. Una sola semilla de entrenamiento por configuración.

**Comparación de las dos configuraciones** (única diferencia: `reward_scale`; métricas de entrenamiento, última ventana de 100 000 pasos):

| | Config 1 (`reward_scale = 1`) | Config 2 (`reward_scale = 0.02`) |
|---|---|---|
| A 2 M de pasos: banderas | 0 de 177 | 63 de 462 (13.6 %) |
| A 5 M de pasos: banderas | 0 de 3333 (muere en x ≈ 314) | 348 de 368 (94.6 %) |
| Tiempo de entrenamiento | ≈ 2.7 h | ≈ 2.8 h |

**Agente final: `best.zip` (checkpoint de 3 800 896 pasos de la Config 2)**, evaluado con semillas que no se usaron para elegirlo:

| Prueba | Resultado |
|---|---|
| P1: inicio estándar, 100 episodios estocásticos (semillas 1000 a 1099) | 99 de 100 llegan a la bandera (IC 95 %: 94.6 a 99.8 %), mediana de 1082 fotogramas |
| P2: con 0 a 32 fotogramas de retraso inicial, 100 episodios estocásticos | 96 de 100 (90.2 a 98.4 %) |
| Determinista, 31 retrasos iniciales (0 a 120 fotogramas) | 31 de 31 (89.0 a 100 %) |
| Determinista, inicio estándar | bandera en 1080 fotogramas |

El agente portable coincide exactamente con SB3 (bandera, x máxima y fotogramas) en 62 episodios deterministas, y produce las mismas acciones en tres entornos (WSL, Windows y Colab). El detalle, el análisis de errores y la selección del checkpoint están en `notebooks/notebook_mario.ipynb`.

## Estructura del repositorio

```
├── README.md
├── requirements.txt            dependencias de entrenamiento (numpy<2, opencv<5)
├── requirements-lock.txt       pip freeze del entorno de Linux (referencia; no sirve en Windows ni Colab)
├── requirements-agent.txt      agente entregable: torch, numpy, opencv-python-headless
├── requirements-play.txt       jugar en vivo con el emulador, sin SB3
├── configs/                    base.json (Config 1), cfg2_reward_scale.json (Config 2) y pilotos
├── src/mario_agent/
│   ├── agent.py  policy.py  preprocess.py  actions.py     agente portable (torch, numpy, opencv)
│   └── wrappers.py  env_factory.py  callbacks.py  export.py   entrenamiento (necesitan gym y SB3)
├── scripts/                    entrenamiento, evaluación, videos, diagnósticos y pruebas de portabilidad
├── notebooks/notebook_mario.ipynb
├── models/final/               agente final: policy_3800896.pt (exportado) y su modelo original de SB3
└── results/                    CSV de las corridas, evaluaciones y línea base aleatoria
```

Los checkpoints intermedios, `runs/` (TensorBoard) y los videos no se versionan (`.gitignore`).

## Entorno y dependencias

| Uso | Dónde se probó | Versiones |
|---|---|---|
| Entrenamiento y evaluación | PC con WSL2 (Ubuntu 22.04) | Python 3.10.12, torch 2.14.1+cu130, stable-baselines3 2.9.2, gymnasium 1.4.0, gym 0.26.2, nes-py 8.2.1, gym-super-mario-bros 7.4.0, numpy 1.26.4, opencv-python-headless 4.11.0.86, pandas 2.3.3, matplotlib 3.10.9 |
| Agente (sin emulador) | Windows, solo CPU | Python 3.14.10, torch 2.15.0+cpu, numpy 2.5.5, opencv 5.1.0 |
| Agente (sin emulador) | Google Colab, solo CPU | Python 3.13, torch 2.14.1+cpu, numpy 2.4.6, opencv 5.0.0 |

`numpy<2` y `opencv<5` solo son necesarios para **entrenar** (por `nes-py`); el agente funciona con NumPy 2 y OpenCV 5. La ROM del juego viene incluida en el paquete `gym-super-mario-bros`; **este repositorio no distribuye ninguna ROM**.

## Instalación

### Entrenar y evaluar (Linux o WSL2 con Ubuntu 22.04, Python 3.10)

```bash
sudo apt update && sudo apt install -y build-essential python3-dev python3.10-venv
git clone https://github.com/Kenauh/mario-ai-ppo.git && cd mario-ai-ppo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_pipeline.py        # comprueba el puente al emulador, los wrappers y los entornos paralelos
```

`train.py` usa la GPU si está disponible (`--device cpu` para entrenar sin GPU, más lento).

### Solo el agente (cualquier sistema, sin emulador)

```bash
python -m venv .venv && source .venv/bin/activate     # en Windows: py -m venv .venv  y  .venv\Scripts\python.exe
pip install -r requirements-agent.txt
python scripts/check_fixture.py models/final/policy_3800896.pt frames_actions.npz   # prueba sin emulador (ver más abajo)
```

### Jugar en vivo con el emulador, sin SB3

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-play.txt
python scripts/play.py models/final/policy_3800896.pt --out partida.mp4
# esperado: causa=bandera | max_x=3161 | fotogramas=1080 | flag_get=True
```

## Usar el agente

El agente recibe **un fotograma RGB crudo (240×256×3, uint8) por llamada**, mantiene su propio buffer de 4 fotogramas, decide cada 4 llamadas y repite la acción entre decisiones. La política está congelada: no se modifica durante el intento.

```python
from mario_agent.agent import MarioAgent

agent = MarioAgent("models/final/policy_3800896.pt", device="cpu", deterministic=True, seed=0)
agent.reset()                      # reiniciar el estado interno al empezar un intento
action = agent.act(obs)            # obs: fotograma RGB crudo -> índice de acción (0 a 6)
names = agent.buttons[action]      # lista de botones, p. ej. NOOP o right+A
agent.close()                      # finalizar
```

| Interfaz común del docente | En este agente |
|---|---|
| Inicializar | `MarioAgent(modelo.pt, ...)` |
| Reiniciar su estado interno | `agent.reset()` |
| Recibir observaciones | `agent.act(obs)` con un fotograma RGB crudo |
| Obtener una acción válida | índice entre 0 y 6 (SIMPLE_MOVEMENT: NOOP, right, right+A, right+B, right+A+B, A, left); `agent.buttons` da los botones |
| Finalizar | `agent.close()` |

Modo determinista (`deterministic=True`, la acción más probable) o estocástico (`deterministic=False`, muestrea con la semilla dada). **Si el evaluador entrega una observación por decisión** (ya con el salto de fotogramas aplicado), crear el agente con `decide_every=1`. **Si entrega otro formato** (gris, 84×84 o apilado), hace falta un adaptador: el agente espera RGB uint8 de forma (H, W, 3) y lanza `ValueError` en otro caso. `reset()` reinicia también el generador aleatorio con la semilla del constructor, así que en modo estocástico dos intentos sobre el mismo emulador dan el mismo resultado; para intentos distintos hay que crear el agente con otra semilla.

El adaptador al evaluador del docente se ajustará cuando llegue el anexo técnico; lo único que dependería de los controles es la lista de botones de `src/mario_agent/actions.py`.

## Reproducir los experimentos

**Entrenamiento** (tiempos medidos en una RTX 4070 Ti; 2 M de pasos ≈ 1.2 h):

```bash
# Config 2 (reward_scale = 0.02): 2 M de pasos y luego continuación hasta 5 M
python scripts/train.py --config configs/cfg2_reward_scale.json
python scripts/train.py --config configs/cfg2_reward_scale.json --resume --total-steps 5000000
# Config 1 (reward_scale = 1): 5 M de pasos
python scripts/train.py --config configs/base.json --total-steps 5000000
python scripts/monitor.py cfg2_reward_scale        # progreso de una corrida frente al azar
```

Cada corrida escribe en `runs/<nombre>/` (episodes.csv, progress.csv, eval.csv, `run_info_*.json` con el comando, las versiones, la GPU y el commit) y en `models/<nombre>/`. En mis corridas, cfg1 se interrumpió a los 2 603 008 pasos por un reinicio de Windows y se reanudó desde el checkpoint de 2 600 000; sus CSV tienen filas duplicadas en ese tramo.

**Evaluación estandarizada** (checkpoints de 2 M y 5 M, protocolos P1 y P2, intervalos de Wilson):

```bash
bash scripts/run_eval.sh                       # escribe results/eval/*.csv
python scripts/eval_standard.py --ckpt models/final/best_sb3_3800896.zip --label cfg2_best --mode det \
    --episodes 31 --noop-max 30 --out results/eval/test_cfg2_best_det_n30.csv
python scripts/eval_agent.py --model models/final/policy_3800896.pt --label best --mode det \
    --episodes 31 --noop-max 30 --compare results/eval/test_cfg2_best_det_n30.csv --out results/eval/agent_best_det_n30.csv
```

`eval_standard.py` evalúa el modelo de SB3; `eval_agent.py` evalúa el agente portable sobre fotogramas crudos y, con `--compare`, lo contrasta episodio por episodio con el de SB3.

**Selección del checkpoint final:** `scripts/run_select.sh` (candidatos tardíos, semillas 0 a 99). Regla fijada antes de ver resultados: un candidato reemplaza al de 5 M solo si logra al menos 87 banderas de 100; con diferencia de 2 puntos o menos, gana la menor mediana de fotogramas. El elegido se reporta con semillas de prueba 1000 a 1099 (`--seed-start 1000`).

**Cadena de Markov por zonas de x:**

```bash
python scripts/markov_zones.py results/cfg2_reward_scale_5M/episodes.csv --since 1500000 --until 2000896
```

**Videos:** `scripts/watch.py` (checkpoint de SB3, con `--stochastic`, `--seed` y `--noop-k` para el retraso inicial) y `scripts/play.py` (modelo `.pt`).

## Prueba de portabilidad sin emulador

`scripts/make_fixture.py` graba, en la máquina con emulador, una secuencia de fotogramas crudos y las acciones que tomó el agente. En otra máquina (sin emulador, SB3 ni Gym), `scripts/check_fixture.py` le da esos fotogramas al agente y comprueba que produce las mismas acciones y que el modelo tiene el mismo `sha256`. La grabación (`fixture/`) no se versiona porque contiene fotogramas del juego. `scripts/test_parity.py` es una prueba rápida de paridad entre SB3 y el agente con una red sin entrenar; la paridad con el modelo final está en `scripts/eval_agent.py` (ver arriba).

## Notebook

`notebooks/notebook_mario.ipynb` documenta las nueve partes pedidas (representación, estrategia, configuración inicial, entrenamiento, segunda configuración, resultados, comparación, análisis de errores y selección del agente final). Lee los CSV de `results/`, no vuelve a entrenar, y funciona en local y en Colab. Las celdas que necesitan el emulador (entrenamiento corto y jugar un episodio) están protegidas con una bandera y mensajes de ayuda.

## Recursos utilizados

Entrenamiento: PC con Ryzen 7 5700G, 16 GB de RAM, RTX 4070 Ti (12 GB), Windows 11 con WSL2. Aproximadamente 5.5 h de entrenamiento formal (cfg1 ≈ 2.7 h, cfg2 ≈ 2.8 h) más las evaluaciones (unos minutos por checkpoint). Pruebas de portabilidad: laptop con Ryzen 9 270 (Windows, solo CPU) y Google Colab (CPU).

## Licencias y uso de herramientas

- No se distribuye ninguna ROM ni archivo del juego. El emulador y el entorno vienen de los paquetes `nes-py` y `gym-super-mario-bros`; cada biblioteca conserva su licencia.
- Se usó un asistente de IA generativa (Claude, de Anthropic) como apoyo técnico para escribir y depurar código y para revisar resultados. Las decisiones de diseño y los experimentos los ejecuté y los valido yo, y soy responsable de explicarlos y defenderlos.

## Limitaciones y pendientes

- **Parámetros del evaluador sin confirmar** (esperan el anexo técnico del docente): límite de 8000 fotogramas, un intento igual a una vida, observación RGB de 240×256×3, salto de 4 fotogramas, 7 acciones, x de la bandera = 3161 (medida aquí) y fórmula de normalización.
- Una semilla de entrenamiento por configuración; la comparación solo varía la escala de recompensa y no aísla el mecanismo.
- La única perturbación de robustez probada es el retraso inicial con NOOP.
- La portabilidad se midió con una trayectoria de unas 150 decisiones, comparando acciones y no logits.
