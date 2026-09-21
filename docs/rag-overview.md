# Как устроены RAG и Chroma в AgentSmith

RAG — не один вызов модели, а две связанные системы. Indexing подготавливает
поисковое представление документов, а query-time pipeline находит evidence и только
после этого формирует ответ.

## Два потока данных

```text
INDEXING (заранее)                         QUERY-TIME (на каждый вопрос)

Markdown policies                         User question
       │                                        │
       ▼                                        ▼
LangChain Document                        query embedding
content + provenance                           │
       │                                        ▼
       ▼                                  Chroma top-k search
text splitter                                  │
       │                                        ▼
       ▼                                  chunks + metadata + distance
chunks + stable IDs                            │
       │                                        ▼
       ▼                                  grade evidence
document embeddings                       ┌─────┴─────┐
       │                            enough │           │ insufficient
       ▼                                   ▼           ▼
Chroma collection                    cited answer   abstention
```

Изменение source document требует повторной индексации. Изменение вопроса — нет.
Embedding function при записи и чтении должна быть одной и той же: другая
размерность технически несовместима, а другая semantic model с той же размерностью
может молча испортить ranking.

## Что хранит Chroma

Каждый record связывает четыре части:

```text
stable chunk ID
├── исходный текст
├── embedding vector
└── metadata
    ├── source
    ├── department
    ├── policy_type
    └── chunk_index
```

В уроках 14–15 коллекция существует только в памяти. Урок 16 задаёт
`persist_directory`, открывает тот же индекс повторно и показывает add, update,
upsert и delete. `.agentsmith/chroma/` — производный артефакт, а не источник истины;
его безопасно удалить и собрать заново из `knowledge/`.

Stable IDs выводятся из source, позиции и текста chunk. Поэтому повторная запись
неизменённого корпуса не размножает records. Полная синхронизация произвольного
внешнего corpus намеренно не рассматривается.

## Почему offline embeddings выглядят необычно

`EducationalEmbeddings` использует небольшой набор видимых concept dimensions:
vacation, parental leave, remote work, security, lost devices, secrets, expenses,
travel и meals. Synonyms вроде `holiday` и `vacation` попадают в одну dimension,
после чего vector нормализуется.

Это позволяет буквально увидеть причину близости:

```text
"annual holiday"  -> [1, 0, 0, ...]
"vacation days"   -> [1, 0, 0, ...]
"expense receipt" -> [0, 0, 0, ..., 1, ...]
```

Такая модель детерминирована и не скачивает weights, но не понимает общий язык и не
подходит production search. `--live` переключает урок на OpenAI embeddings и может
отправить документные chunks провайдеру.

## Retrieval controls

- `k` ограничивает число кандидатов и объём context window.
- Metadata filter исключает records до или во время vector search, например оставляет
  только `department=finance`.
- Similarity search выбирает ближайшие vectors.
- MMR балансирует relevance и различие между выбранными chunks.
- Distance зависит от embedding model и metric. Это ranking signal, не универсальная
  вероятность правильности.

Threshold в примерах настроен только для educational embeddings. Его нельзя без
измерений переносить на другую модель или corpus.

## Три control-flow варианта

```text
Two-step RAG       question -> retrieve -> generate/abstain
Agentic RAG        question -> model -> [retrieve_docs?] -> model
LangGraph RAG      retrieve -> grade -> generate
                                  └──-> abstain
```

Two-step вариант проще и всегда обращается к knowledge base. Agentic вариант может
не делать лишний поиск для обычного приветствия, но решение модели становится ещё
одной измеряемой точкой отказа. LangGraph вариант явно хранит documents, distances,
grade, answer, sources и пройденный route.

Retrieved text передаётся как ограниченные `<evidence>` blocks с source. Генератору
предписано считать их данными, а не инструкциями. Ответ может цитировать только
фактически переданные sources. Если threshold не пройден, workflow возвращает
фиксированное abstention вместо догадки.

## Evaluation: где именно ошибка

Один «правильный ответ» скрывает разные дефекты. Урок 20 считает отдельно:

| Metric | Вопрос |
|---|---|
| retrieval hit@k | Попал ли ожидаемый source в top-k? |
| grounding | Поддерживается ли ответ evidence? |
| citation validity | Были ли cited sources среди retrieved? |
| abstention | Отказалась ли система на unsupported case? |
| expected tool use | Вызвал ли agent retrieval только когда требовалось? |

Правдоподобный текст при неверном source остаётся retrieval failure. Точно так же
правильный source не оправдывает выдуманный ответ или ложную citation.

## Privacy и типичные сбои

- Не помещайте реальные внутренние документы или credentials в учебный corpus.
- Live embeddings и chat generation отправляют данные provider; tracing может
  отправить ещё и полную trajectory.
- Слишком маленькие chunks теряют контекст, слишком большие ухудшают точность и
  расходуют context window.
- Дубли обычно означают нестабильные IDs или неявный lifecycle collection.
- Плохой ranking сначала диагностируют по chunks, metadata и retrieved sources, а не
  маскируют более убедительным prompt.
- Prompt injection из retrieved documents — отдельная production-тема; delimiters и
  evidence-only instruction снижают риск, но не являются полной защитой.
