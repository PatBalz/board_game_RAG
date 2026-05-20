# Dune Imperium Rules Assistant (Local RAG, CPU-only)

This project is a proof-of-concept **retrieval-augmented generation (RAG)** assistant for understanding the rules of **_Dune: Imperium_**.  
The key goal is that the system can run **fully locally on a CPU-only machine** (no external embedding/LLM APIs required).

## What it does
- Builds a searchable rules knowledge base from locally stored documents.
- Uses a retriever to find the most relevant rule passages for a given question.
- Uses a local generator model to answer, grounded in the retrieved text.

## Features
- **CPU-only friendly** setup
- **Local indexing + retrieval** (runs offline after setup)
- Answers with **citations to retrieved rule text** (recommended/implemented depending on configuration)

## Repository layout (target-picture ;))
- `data/` – source documents for Dune Imperium rules/text (your files)
- `src/` – RAG pipeline code (ingest/index/retrieve/generate)
- `config/` – model + pipeline configuration
- `notebooks/` – experiments and evaluation helpers (optional)

## Notes / Design goals
- Retrieval quality matters: chunking, overlap, and metadata (page/section) will be tuned to find optimum
- For best results, the generator should be instructed to:
  - use retrieved passages as the source of truth
  - acknowledge uncertainty when retrieval is insufficient

## Next steps (roadmap)
- Improve chunking strategy for complex rules lookups
- Implement lightweight evaluation (coverage of common rules queries)
- Expand beyond text rules (e.g., glossary of terms, edge-case handling)
- Maybe implement a reinforcement training via skilled users


