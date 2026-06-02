# Движок инференса ONNX (coreEngine.py)

## Назначение

Модуль `coreEngine.py` предоставляет абстрактный базовый класс `EngineBase` и его реализацию `OnnxEngine` для унифицированного запуска нейросетевого вывода через ONNX Runtime. Все детекторы (YOLO, EfficientDet, UFLD/UFLDv2) используют `OnnxEngine` как единый интерфейс доступа к моделям.

---

## Иерархия классов

```
EngineBase (abc.ABC)
    │   get_engine_input_shape()   → форма входного тензора
    │   get_engine_output_shape()  → форма выходных тензоров
    │   engine_inference()         → запуск вывода
    │
    └── OnnxEngine
            session: onnxruntime.InferenceSession
            engine_dtype: np.float16 | np.float32
            framework_type: "onnx"
```

---

## Код: EngineBase

Абстрактный класс задаёт контракт для любого инференс-движка.

```python
# coreEngine.py:8-38

class EngineBase(abc.ABC):
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The model path [%s] can't be found!" % model_path)
        assert model_path.endswith('.onnx'), 'Model path must be a .onnx file.'
        self._framework_type = None

    @property
    def framework_type(self):
        if self._framework_type is None:
            raise Exception("Framework type can't be None")
        return self._framework_type

    @abc.abstractmethod
    def get_engine_input_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def get_engine_output_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def engine_inference(self):
        return NotImplemented
```

**Ключевые моменты:**
- Конструктор проверяет существование файла и расширение `.onnx`.
- Свойство `framework_type` защищено от обращения до явной установки.

---

## Код: OnnxEngine

Конкретная реализация для ONNX Runtime с автоматическим выбором провайдера (GPU/CPU).

```python
# coreEngine.py:55-100

class OnnxEngine(EngineBase):
    def __init__(self, onnx_file_path, num_threads=None):
        EngineBase.__init__(self, onnx_file_path)

        sess_opts = _create_session_options(num_threads)

        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.session = onnxruntime.InferenceSession(
                onnx_file_path, sess_opts, providers=['CUDAExecutionProvider']
            )
            print("[INFO] ONNX Runtime using GPU (CUDA)")
        else:
            if num_threads is None:
                half_cores = max(1, os.cpu_count() // 2)
                sess_opts.intra_op_num_threads = half_cores
                sess_opts.inter_op_num_threads = 1
            self.session = onnxruntime.InferenceSession(
                onnx_file_path, sess_opts, providers=['CPUExecutionProvider']
            )
            print(f"[INFO] ONNX Runtime using CPU (intra_op_threads={sess_opts.intra_op_num_threads})")

        self.providers = self.session.get_providers()
        self.engine_dtype = (
            np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        )
        self.framework_type = "onnx"
        self.__load_engine_interface()
```

**Ключевые моменты:**
- Приоритет отдаётся `CUDAExecutionProvider`; при его отсутствии используется `CPUExecutionProvider`.
- Если `num_threads` не задан, автоматически используется половина доступных ядер CPU.
- Тип данных (`float16`/`float32`) определяется из метаданных модели.

---

## Код: Настройка сессии

```python
# coreEngine.py:40-52

def _create_session_options(num_threads=None):
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    if num_threads is not None and num_threads > 0:
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
    else:
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 0
    return opts
```

| Параметр | Значение | Описание |
|----------|----------|----------|
| `graph_optimization_level` | `ORT_ENABLE_ALL` | Включает все оптимизации графа ONNX |
| `enable_mem_pattern` | `True` | Оптимизация шаблонов доступа к памяти |
| `enable_mem_reuse` | `True` | Повторное использование памяти между запусками |
| `execution_mode` | `ORT_SEQUENTIAL` | Последовательное выполнение (без параллелизма внутри одного инференса) |

---

## Вызов инференса

```python
# coreEngine.py:96-100

def engine_inference(self, input_tensor):
    output = self.session.run(
        self.__output_names, {self.__input_names[0]: input_tensor}
    )
    return output
```

- На вход подаётся тензор в формате, соответствующем `engine_dtype` (`float32` или `float16`).
- Метод возвращает список выходных тензоров `output` — каждый Ergebnis вызова нейросети.

---

## Интеграция с детекторами

Каждый детектор использует `OnnxEngine` следующим образом:

```python
# Общий паттерн (пример из YoloDetector)
self.engine = OnnxEngine(model_path, num_threads=self._num_threads)
self.set_input_details(self.engine)   # → input_shapes, input_types
self.set_output_details(self.engine)  # → output_shapes, output_names

# При вызове DetectFrame:
output = self.engine.engine_inference(input_tensor)[0]
```

Аналогично `UltrafastLaneDetectorV2`:
```python
self.engine = OnnxEngine(model_path, num_threads=self._num_threads)
output = self.engine.engine_inference(input_tensor)
```

---

## Диаграмма использования

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐
│ YoloDetector │    │ Efficientdet │    │ UltrafastLaneDetV2   │
│              │    │   Detector   │    │                     │
└──────┬───────┘    └──────┬───────┘    └──────────┬──────────┘
       │                   │                       │
       └───────────────────┼───────────────────────┘
                           │
                    ┌──────▼───────┐
                    │  OnnxEngine  │
                    │  (coreEngine)│
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  onnxruntime.InferenceSession │
              │  (GPU CUDA / CPU)            │
              └──────────────────────────────┘
```

---

## Связанные разделы

- [Детекция объектов](object-detection.md)
- [Детекция разметки](lane-detection.md)
- [Архитектура решения](architecture.md)