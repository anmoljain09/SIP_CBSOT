# 📚 Research Paper Intelligence System

> Ask your research corpus questions in plain English — and get back search results, summaries, keywords, and comparisons.

Made by **Anmol Jain**

---

## 🎯 What This Is

Reading through hundreds of ML papers to find the ones that matter is slow. This project builds a small "intelligence layer" on top of a paper corpus so you can just *ask*:

- "What's out there on vision transformers?"
- "Summarize the top papers on graph neural networks."
- "How do ViTs compare to CNNs?"

...and get grounded, relevant answers — not hallucinated ones — because every answer is backed by an actual retrieved paper from the dataset.

## ⚙️ Core Capabilities

| Capability | How it's done |
|---|---|
| 🔎 Semantic search | `sentence-transformers` embeddings + a `FAISS` vector index |
| 🧾 Summarization | `facebook/bart-large-cnn`, with output length adapted to input length |
| 🏷️ Keyword extraction | `KeyBERT` keyphrase extraction |
| 🧠 Entity recognition | General-purpose `transformers` NER pipeline (models, datasets, institutions) |
| ⚖️ Paper comparison | LLM-generated structured comparison across objective, methodology, contributions, advantages, limitations, applications |
| 💬 Natural-language interface | A LangChain agent (Llama 3.1 8B via Groq) that automatically picks the right tool for a given question |

## 🧩 Pipeline at a Glance

```
Raw papers (title + abstract)
        │
        ▼
   Clean & dedupe
        │
        ▼
 Sentence-Transformer embeddings (all-MiniLM-L6-v2)
        │
        ▼
   FAISS vector index  ──►  Semantic search
        │                         │
        ▼                         ▼
  BART summarization      KeyBERT keywords + NER
        │                         │
        └────────────┬────────────┘
                      ▼
        LangChain tools + Llama 3.1 agent
                      │
                      ▼
        Natural-language Q&A over your corpus
```

## 📦 Built With

- **Data**: [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face) — a sample of up to 15,000 ML paper title+abstract pairs from arXiv
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector search**: `faiss-cpu`
- **Summarization / NER**: `transformers`
- **Keywords**: `keybert`
- **Agent orchestration**: `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface`
- **LLM**: `langchain-groq` → Llama 3.1 8B Instant
- **Data wrangling**: `pandas`, `numpy`, `scikit-learn`

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install datasets sentence-transformers faiss-cpu keybert \
    transformers==4.46.3 huggingface_hub==0.26.2 tokenizers==0.20.3 sentence-transformers==3.3.1 \
    langchain langchain-community langchain-core langchain-huggingface langchain-groq
```

### 2. Add your Groq API key

The agent section needs an LLM. Grab a free key from [console.groq.com](https://console.groq.com), then:

```bash
export GROQ_API_KEY="your-api-key-here"
```

(Running in Colab? The notebook will pick it up from Colab's secrets manager automatically.)

### 3. Run the notebook

Open `Research_Paper_Intelligence_System_1.ipynb` and run the cells in order — each section builds on the objects (`df`, `embed_model`, `index`, `llm`, `agent`) created in the ones before it.

### 4. Query it

Directly, in Python:

```python
display_search_and_summarize("deep learning for medical image analysis", k=5)
```

Or conversationally, through the agent:

```python
agent.invoke({
    "messages": [
        {"role": "user", "content": "Compare a paper on Vision Transformers with a paper on CNNs."}
    ]
})
```

## 🗂️ What's in the Repo

```
.
├── Research_Paper_Intelligence_System_1.ipynb   → the whole pipeline, notebook-first
├── paper_embeddings.npy                         → cached embeddings (auto-generated)
└── paper_faiss.index                            → cached FAISS index (auto-generated)
```

## 🔮 Where This Could Go Next

- Swap in a paper-specific embedding model like SPECTER2 for sharper retrieval
- Move to an approximate FAISS index (IVF / HNSW) to handle much larger corpora
- Filter results by year, category, or venue
- Wrap it in a small FastAPI or Streamlit app for a proper front end

## ✍️ Author

**Anmol Jain**
