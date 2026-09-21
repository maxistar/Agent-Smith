# AgentSmith: LangGraph шаг за шагом

Учебный Python-проект, который показывает, как обычный вызов LLM постепенно превращается в агента с инструментами, состоянием и оценкой качества, а затем — в RAG-систему с Chroma.

Каждый урок запускается отдельно и добавляет одну основную идею. По умолчанию используется детерминированная офлайн-модель: API-ключи, сеть и платные вызовы не нужны. Флаг `--live` явно включает настоящий OpenAI-вызов.

Подробная ментальная модель всей системы находится в [docs/system-overview.md](docs/system-overview.md), а indexing и query-time RAG разобраны в [docs/rag-overview.md](docs/rag-overview.md).

## Быстрый старт

Требования:

- Python 3.14;
- [uv](https://docs.astral.sh/uv/);
- OpenAI API key только для запусков с `--live`;
- LangSmith API key только для live-трассировки в уроке 08.

Установка и первый безопасный запуск:

```bash
uv sync
uv run python -m examples.lesson_01_model_call
uv run pytest
```

Все команды выполняются из корня репозитория.

## Учебная последовательность

| № | Урок | Что наблюдаем | Команда |
|---|---|---|---|
| 01 | Model call | `HumanMessage` на входе и `AIMessage` на выходе | `uv run python -m examples.lesson_01_model_call` |
| 02 | Direct tool | Обычный вызов Python-tool без LLM | `uv run python -m examples.lesson_02_direct_tool` |
| 03 | Tool request | Модель формирует `tool_calls`, но ничего не исполняет | `uv run python -m examples.lesson_03_tool_request` |
| 04 | Manual loop | Приложение исполняет tool и добавляет `ToolMessage` | `uv run python -m examples.lesson_04_manual_tool_loop` |
| 05 | LangGraph loop | Тот же цикл как nodes, edges и routing | `uv run python -m examples.lesson_05_langgraph_loop` |
| 06 | Conversation state | История сообщений передаётся в следующий turn | `uv run python -m examples.lesson_06_conversation_state` |
| 07 | Persistence | `InMemorySaver` разделяет состояние по `thread_id` | `uv run python -m examples.lesson_07_persistence` |
| 08 | Tracing | LangSmith наблюдает уже работающий граф | `uv run python -m examples.lesson_08_tracing` |
| 09 | Reliability | Ошибки tools и бесконечный цикл становятся ограниченными | `uv run python -m examples.lesson_09_reliability` |
| 10 | Evaluation | Dataset сравнивает правильность ответа и tool use | `uv run python -m examples.lesson_10_evaluation` |
| 11 | Documents | Синтетический корпус и provenance в `Document.metadata` | `uv run python -m examples.lesson_11_documents` |
| 12 | Chunking | Влияние размера, overlap и границ на chunks | `uv run python -m examples.lesson_12_chunking` |
| 13 | Embeddings | Объяснимые vectors и semantic similarity | `uv run python -m examples.lesson_13_embeddings` |
| 14 | Chroma | In-memory индекс и similarity search со scores | `uv run python -m examples.lesson_14_chroma` |
| 15 | Retrieval | `k`, metadata filters и MMR | `uv run python -m examples.lesson_15_retrieval` |
| 16 | Persistence | Rebuild, CRUD и повторное открытие Chroma | `uv run python -m examples.lesson_16_persistence --rebuild` |
| 17 | Two-step RAG | Всегда retrieve, затем cited answer или abstention | `uv run python -m examples.lesson_17_two_step_rag` |
| 18 | Agentic RAG | Модель решает, вызывать ли `retrieve_docs` | `uv run python -m examples.lesson_18_agentic_rag` |
| 19 | LangGraph RAG | Явные retrieve/grade/generate/abstain nodes | `uv run python -m examples.lesson_19_langgraph_rag` |
| 20 | RAG evaluation | Retrieval и answer metrics считаются отдельно | `uv run python -m examples.lesson_20_rag_evaluation` |

### Ключевая разница уроков 03–05

```text
03: LLM -> tool_call                    (только запрос)

04: LLM -> Python executes tool
          -> ToolMessage -> LLM         (ручной цикл)

05: START -> agent -> tools -> agent
                    \-> END             (тот же цикл в графе)
```

LLM не вызывает Python-функцию самостоятельно. Она возвращает структурированное намерение. Исполняет его приложение; LangGraph формализует состояние и переходы.

### RAG quick start

Уроки 11–20 используют только синтетические файлы из `knowledge/`. По умолчанию
работает небольшая объяснимая embedding-модель: сеть, API keys и Chroma Cloud не нужны.

```bash
uv sync
uv run python -m examples.lesson_11_documents
uv run python -m examples.lesson_14_chroma
uv run python -m examples.lesson_17_two_step_rag
uv run python -m examples.lesson_20_rag_evaluation
```

Урок 16 создаёт локальный индекс в `.agentsmith/chroma/`:

```bash
uv run python -m examples.lesson_16_persistence --rebuild
rm -rf .agentsmith/chroma
```

Каталог является генерируемым и игнорируется Git. Удаляйте только этот путь из корня
проекта; индекс всегда можно восстановить из учебного корпуса.

## Live-режим

Скопируйте шаблон и замените placeholder локально:

```bash
cp .env.example .env
```

Файл `.env` игнорируется Git, но проект намеренно не загружает его автоматически. Экспортируйте переменные в текущий shell:

```bash
set -a
source .env
set +a
```

После этого добавьте `--live` к урокам, использующим модель:

```bash
uv run python -m examples.lesson_01_model_call --live
uv run python -m examples.lesson_03_tool_request --live
uv run python -m examples.lesson_04_manual_tool_loop --live
uv run python -m examples.lesson_05_langgraph_loop --live
uv run python -m examples.lesson_06_conversation_state --live
uv run python -m examples.lesson_07_persistence --live
uv run python -m examples.lesson_10_evaluation --live
```

`AGENTSMITH_MODEL` выбирает chat model; значение по умолчанию — `gpt-5-mini`.
`AGENTSMITH_EMBEDDING_MODEL` отдельно выбирает embedding model; по умолчанию
`text-embedding-3-small`. Сейчас учебные factory поддерживают
`AGENTSMITH_PROVIDER=openai`.

Для сетевого RAG-запуска добавьте `--live`, например:

```bash
uv run python -m examples.lesson_13_embeddings --live
uv run python -m examples.lesson_17_two_step_rag --live
```

Live embeddings отправляют chunks и запросы внешнему provider и могут стоить денег.
Offline и live indexes несовместимы: при смене embedding model перестройте индекс.

## LangSmith tracing

Урок 08 без аргументов остаётся офлайн и ничего не отправляет наружу. Для настоящего trace задайте:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="agentsmith-learning-path"
uv run python -m examples.lesson_08_tracing --live
```

Функциональный результат графа не меняется. В LangSmith появляются отдельные шаги model/tool, их входы, выходы, latency, ошибки и доступная token usage.

## Проверки

Default suite полностью офлайн и не требует credentials:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Тесты используют scripted/rule-based model doubles и проверяют:

- известные, неизвестные и некорректные запросы к `search_docs`;
- связь `tool_call_id` между `AIMessage` и `ToolMessage`;
- graph routing и нормальное завершение;
- продолжение и изоляцию threads;
- ограничение повторяющихся tool calls;
- отсутствие требования OpenAI/LangSmith credentials в offline-режиме;
- все четыре evaluation-сценария.
- загрузку и chunking синтетического корпуса;
- embeddings, Chroma search/filters/MMR и persistent reopen;
- citations, abstention и обе ветви RAG-графа;
- retrieval hit@k отдельно от grounding, citations и tool use.

Live smoke checks выполняются отдельно и могут стоить денег:

```bash
uv run python -m examples.lesson_01_model_call --live
uv run python -m examples.lesson_03_tool_request --live
uv run python -m examples.lesson_08_tracing --live
```

## Ограничения и безопасность

- Не коммитьте `.env`, API keys или содержимое реальных внутренних документов.
- Live-вызовы отправляют prompts и tool results внешнему provider; tracing дополнительно отправляет trajectory в LangSmith.
- LLM-ответы недетерминированы, поэтому offline-тесты проверяют структуру, а не точный live-текст.
- `InMemorySaver` из урока 07 сохраняет thread state только внутри текущего процесса. После перезапуска Python состояние исчезает.
- `search_docs` — учебный literal lookup над словарём; `retrieve_docs` — отдельный semantic retrieval tool над Chroma.
- Учебные embeddings прозрачны и детерминированы, но не являются production semantic model.
- Retrieved content считается недоверенными данными: генератор получает его в явных evidence-блоках, однако полноценная защита от prompt injection остаётся вне этого курса.

## Частые проблемы

- **`OPENAI_API_KEY is missing`:** вы запустили урок с `--live`. Экспортируйте ключ
  или уберите флаг для офлайн-режима.
- **`LANGSMITH_TRACING is not enabled`:** для live-урока 08 нужны одновременно
  `LANGSMITH_TRACING=true` и `LANGSMITH_API_KEY`.
- **Модель не вызывает tool:** tool use остаётся решением модели. Проверьте поддержку
  tool calling выбранной моделью и формулировку вопроса. Для повторяемого результата
  запустите без `--live`.
- **State пропал после перезапуска:** это ожидаемое ограничение `InMemorySaver`;
  durable database persistence не входит в этот учебный путь.
- **Цикл остановлен по recursion limit:** защитная граница сработала, потому что модель
  продолжала запрашивать tools вместо финального ответа. См. урок 09.
- **Chroma сообщает о несовместимом embedding:** удалите `.agentsmith/chroma/` и
  запустите урок 16 с `--rebuild`; размерность и смысл vectors должны совпадать.
- **Semantic result выглядит странно:** raw distance не является вероятностью.
  Проверьте chunks, embedding mode, metadata filter и ожидаемый source до настройки threshold.
