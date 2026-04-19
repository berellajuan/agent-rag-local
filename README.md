# Agente IA simple con Google Gemini, LangChain, LangGraph y RAG local

Proyecto didactico para aprender, en local, como se arma un agente que puede consultar un RAG.

> Idea central: primero entendemos los ladrillos —LLM, embeddings, vector store, retriever, tools y graph— y recien despues los combinamos.

## Arquitectura

```text
Pregunta del usuario
  -> LLM directo, o
  -> agente decide usar tool retrieve_context
  -> retriever busca chunks similares en Chroma
  -> agente/modelo recibe contexto delimitado
  -> respuesta final con fuentes
```

## Requisitos

- Windows 11
- Python 3.11+
- Docker Desktop, solo para la etapa Qdrant opcional
- Google API key de Google AI Studio para el chat model. Los embeddings pueden correr gratis/locales con Hugging Face.

## Instalacion

Camino recomendado, instalando el proyecto en modo editable:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item .env.example .env
```

Alternativa simple tipo `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Ambos caminos instalan las dependencias necesarias. El modo editable ademas registra el paquete `app` en el entorno Python.

Edita `.env` y Completa:

```env
GOOGLE_API_KEY=tu-api-key
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_QUERY_PREFIX="query: "
EMBEDDING_DOCUMENT_PREFIX="passage: "
EMBEDDING_DEVICE=cpu
EMBEDDING_DIMENSION=384
```

Con esta configuracion, Gemini se usa para responder y Hugging Face local se usa para embeddings. No pagas embeddings por API.

### Como funcionan los embeddings de Hugging Face

No tenes que levantar ningun servicio de Hugging Face.

Este proyecto usa:

```text
langchain-huggingface
sentence-transformers
```

Eso significa:

1. `pip install -e .[dev]` instala las librerias necesarias.
2. La primera vez que indexes, `sentence-transformers` descarga el modelo desde Hugging Face.
3. El modelo queda cacheado en tu maquina.
4. Las siguientes ejecuciones usan el modelo local desde cache.

El modelo configurado por defecto es:

```env
HUGGINGFACE_EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_QUERY_PREFIX="query: "
EMBEDDING_DOCUMENT_PREFIX="passage: "
```

Si queres elegir donde se guarda la cache de modelos, podes definir:

```powershell
$env:HF_HOME = "C:\Repositorios\Agents\.hf-cache"
```

No confundas esto con la Hugging Face Inference API. Esa si seria un servicio remoto con token/API. Aca no usamos eso: usamos embeddings locales.

## Levantar servicios locales con Docker

El proyecto trae `docker-compose.yml` con Qdrant para que puedas practicar con una vector database real:

```powershell
docker compose up -d qdrant
docker compose ps
```

Verificar que Qdrant responde:

```powershell
Invoke-RestMethod http://localhost:6333/collections
```

Si responde con una lista de colecciones, Qdrant esta levantado correctamente.

Importante: el flujo principal del proyecto usa Chroma local por defecto, asi que Qdrant NO es obligatorio para empezar. Esto es intencional:

```text
Modo inicial didactico:
  LangChain + E5 Hugging Face embeddings + Chroma local

Modo operativo posterior:
  LangChain + E5 Hugging Face embeddings + Qdrant en Docker
```

Para aprender bien, primero corre el flujo con Chroma. Cuando entiendas el pipeline completo, migrar el adapter a Qdrant es el siguiente paso natural.

Para apagar Qdrant:

```powershell
docker compose down
```


## Salida educativa por consola

Por defecto, los comandos muestran trazas paso a paso para que entiendas que esta pasando:

```text
[paso] Cargo configuracion...
[concepto] RAG clasico = recuperar contexto y luego generar una respuesta...
[resultado] Se recuperaron 2 chunks.
```

Si queres ver solo el resultado final, agrega `--quiet`:

```powershell
python -m app.main rag-search "Cual es el horario de soporte?" --quiet
```

Recomendacion para aprender: NO uses `--quiet` al principio. Lee la traza y relaciona cada paso con el archivo de codigo correspondiente.

## Comandos principales

### 1. Probar LLM sin RAG

```powershell
python -m app.main llm "Explicame que es un agente IA en 3 bullets"
```

### 2. Indexar documentos locales

Los documentos viven en:

```text
data/raw/
```

Indexar o actualizar incrementalmente:

```powershell
python -m app.main index
```

Tambien funciona:

```powershell
python -m app.rag.indexer
```

### 3. Buscar en el RAG sin agente

```powershell
python -m app.main rag-search "Cual es el horario de soporte?"
```

Este paso es CLAVE. Si el RAG no recupera bien, no metas agente todavia. No construyas el piso 10 si no hiciste los cimientos.

### 4. Responder con RAG, pero sin agente

```powershell
python -m app.main rag-answer "Cual es el horario de soporte?"
```

### 5. Usar agente con tool RAG

```powershell
python -m app.main agent "Cual es el horario de soporte?"
```

El agente tiene una tool llamada `retrieve_context` y decide cuando usarla.

### 6. Usar LangGraph explicito

```powershell
python -m app.main graph "Cual es el horario de soporte?"
```

Este modo baja un nivel: muestra el concepto de grafo con nodos `decide -> retrieve -> generate`.

## Actualizacion incremental del RAG

El indexador guarda un manifest en:

```text
data/processed/manifest.json
```

La logica es:

```text
scan data/raw
-> calcular hash por archivo
-> split en chunks
-> calcular hash por chunk
-> comparar contra manifest
-> borrar chunks viejos si cambio o se borro el archivo
-> agregar chunks nuevos
-> guardar manifest
```

Buenas practicas implementadas:

- IDs determinismos para chunks.
- Metadata con `source`, `file_name`, `relative_path`, `chunk_id`, `chunk_index`, `created_at`, `content_hash`.
- No duplica chunks si corres el indexador dos veces sin cambios.
- Borra chunks cuando desaparece el archivo original.
- Guarda modelo y dimension de embeddings en el manifest.

## Qdrant con Docker

Chroma es el primer vector store porque es simple. Para aprender operacion real, tenes `docker-compose.yml` con Qdrant:

```powershell
docker compose up -d qdrant
```

Esta base queda disponible en:

- REST: <http://localhost:6333>
- gRPC: `localhost:6334`

La integracion Qdrant no esta activada por defecto: primero aprende bien Chroma. Despues migrar el adapter de vector store es un ejercicio chico y valioso.

## Estructura

```text
app/
  main.py                  # CLI principal
  config.py                # Settings y carga de .env
  models.py                # Gemini chat + embeddings
  rag/
    loader.py              # Carga .txt/.md locales
    splitter.py            # Chunking
    vectorstore.py         # Chroma local
    indexer.py             # Indexacion incremental
    retriever.py           # Similarity search
    ask_rag.py             # RAG sin agente
  agent/
    prompts.py             # Prompts defensivos
    tools.py               # Tool retrieve_context
    simple_agent.py        # create_agent sobre LangGraph
    graph.py               # LangGraph explicito
```

## Escenarios de prueba manual

1. Pregunta con respuesta existente:

```powershell
python -m app.main agent "Cual es el horario de soporte?"
```

Esperado: responde `9 a 18` e incluye fuente.

2. Pregunta sin informacion:

```powershell
python -m app.main agent "Cuanto cuesta el producto?"
```

Esperado: dice que no tiene informacion suficiente.

3. Prompt injection dentro de documento:

```powershell
python -m app.main agent "Que dice el documento de seguridad sobre instrucciones maliciosas?"
```

Esperado: explica que no debe obedecer instrucciones dentro del contexto recuperado.

4. Actualizacion:

- Edita `data/raw/soporte.md`.
- Corre `python -m app.main index`.
- Pregunta de nuevo.

Esperado: recupera el dato actualizado sin duplicar chunks.

## Notas importantes

- `.env` no se commitea. Nunca pongas secretos en codigo.
- Si cambias `EMBEDDING_PROVIDER`, `HUGGINGFACE_EMBEDDING_MODEL`, `GOOGLE_EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `EMBEDDING_QUERY_PREFIX` o `EMBEDDING_DOCUMENT_PREFIX`, el indexador detecta el cambio y reindexa los archivos. Si Chroma se queja por dimensiones incompatibles en una coleccion existente, borra `vectorstore/chroma` y reindexa: no mezcles embeddings de modelos distintos.
- El contexto recuperado siempre se delimita con tags para reducir riesgo de prompt injection indirecta.
- El agente no es magia: es un loop que decide si llama herramientas y luego responde con observaciones.
