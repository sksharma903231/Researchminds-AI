# ResearchMinds-AI Project Handover

**Repository:** /home/sumit/projects/Researchminds-AI  
**Project:** ResearchMinds-AI  
**Current branch:** feature/phase-1-foundation  
**Base/current commit:** a492954b82c17d2c2e2cd94de853973635a85907  
**Handover prepared:** 2026-11-01

This document is the continuation record for an engineering assistant joining without access to earlier conversations. It distinguishes the approved target from the current implementation and records unfinished work explicitly.

## 1. Product overview

ResearchMinds-AI is currently a single-user Streamlit retrieval-augmented generation (RAG) prototype for querying a local collection of academic PDFs. The current code extracts PDF text, splits it into chunks, creates SentenceTransformers embeddings, stores vectors in FAISS, retrieves relevant chunks, and generates responses through either Ollama or Groq.

The approved product direction is to evolve this prototype into a multi-user, cloud-deployable document intelligence platform with general document support. The future architecture is a modular monolith with a web frontend, API, persistent metadata/vector storage, object storage, managed authentication, and background ingestion. This is a target, not work completed in this phase.

## 2. Architecture and engineering decisions

### Approved target architecture

| Area | Approved direction | Current status |
|---|---|---|
| Application shape | Modular monolith | Target; current code is a small Python prototype |
| API | FastAPI | Target; not implemented |
| Frontend | Next.js + TypeScript | Target; not implemented |
| Metadata and vectors | PostgreSQL + pgvector | Target; current vector store is local FAISS |
| File storage | S3-compatible object storage | Target; not implemented |
| Authentication | Managed authentication | Target; not implemented |
| Ingestion | Background ingestion worker | Target; current ingestion is local command-line pipelines |
| Model inference | Local and cloud LLM inference | Existing Ollama and Groq modes are to be preserved |
| Input documents | General document support | Target; current ingestion is PDF-focused |

### Approved engineering decisions

- Python 3.14.
- pip-tools for dependency inputs and generated pinned requirement files.
- pytest and pytest-cov for testing.
- Minimum 80% line coverage for the production package.
- Incremental migration; do not move the whole project in one change.
- Selective LangChain packages rather than relying on the umbrella package.
- Retain the existing three PDFs as demo/test fixtures.
- IntelliJ IDEA is the primary IDE.
- Preserve current Streamlit, Ollama, and Groq functionality during migration.

## 3. Repository audit summary

The repository is on feature/phase-1-foundation at a492954b82c17d2c2e2cd94de853973635a85907. The branch was created from the audited master HEAD. No changes have been staged or committed for WP1.

Relevant current layout:

- app.py, app_public.py: Streamlit entry points.
- main.py: command-line-style entry point.
- src/ingestion/: PDF extraction and chunking pipelines.
- src/vectorstore/: FAISS index construction.
- src/retrieval/: FAISS retrieval and embedding model loading.
- src/generation/: Groq and Ollama response generation.
- src/ui/: Streamlit chat UI.
- data/papers/: three tracked PDF fixtures.
- data/processed/texts/: three tracked extracted-text fixtures.
- data/vectorstore/: tracked papers.index and index_mapping.json.

The current .gitignore excludes .venv/, venv/, env/, generated registries, and data/vectorstore/. The vectorstore files remain tracked even though the ignore rules cover their paths; ignore rules do not untrack files. Do not regenerate or overwrite them without explicit scope and review.

The source imports justify the direct production dependency candidates in requirements.in: Streamlit, PyMuPDF, SentenceTransformers, PyTorch, FAISS, langchain-core, langchain-text-splitters, langchain-groq, and langchain-ollama. Source imports do not justify direct dependencies on the langchain umbrella package or langchain-community.

The repository README still describes the original prototype and lists Python 3.10 or later. The approved foundation target is Python 3.14; README updates have not been part of WP1.

## 4. Environment and Git state

### Git state

- Branch: feature/phase-1-foundation.
- HEAD: a492954b82c17d2c2e2cd94de853973635a85907.
- Existing .idea/ files are untracked and must be preserved.
- WP1 changed .gitignore and replaced the old requirements.txt manifest contents with the generated production lock.
- requirements.in, requirements-dev.in, and requirements-dev.txt are untracked new files.
- No changes are staged or committed.

Always inspect git status --short --branch --untracked-files=all before and after work. Do not stage or commit without review. Preserve all existing untracked .idea/ files.

### Python environments

- New .venv/: Python 3.14.4, pip 25.1.1, pip-tools 7.6.2.
- .venv/pyvenv.cfg says include-system-site-packages = false and identifies /usr/bin/python3.14 as its base interpreter.
- pip-tools is installed in .venv/ only. Production packages, pytest, and pytest-cov have not been installed there.
- The old venv/ has metadata for Python 3.12.3 and is known to be inconsistent. It must remain untouched; do not repair, recreate, or delete it.
- .venv/ is ignored by Git.

The shell may supply a PYTHONPATH for unrelated ROS Python packages. Earlier validation found that clearing PYTHONPATH for venv commands gives an isolated Python path and a clean pip check; use env -u PYTHONPATH when checking or running venv tools if the shell still exports it. Do not modify the global shell or ROS configuration as part of this project.

## 5. Completed work: WP1 environment and dependency foundation

Completed:

1. Confirmed branch feature/phase-1-foundation and base commit.
2. Created .venv/ using Python 3.14.4.
3. Installed pip-tools 7.6.2 inside .venv/ only.
4. Added .venv/ to .gitignore.
5. Reviewed source imports and created requirements.in with the direct production dependencies listed above.
6. Created requirements-dev.in with -r requirements.in, pip-tools, pytest, and pytest-cov.
7. Compiled requirements.txt and requirements-dev.txt using pip-tools under Python 3.14.4. Both locks contain exact version pins and resolution completed without reported dependency conflicts.
8. Confirmed .venv/ isolation when PYTHONPATH is cleared.

Not done:

- Production dependencies have not been installed.
- pytest and pytest-cov have not been installed in .venv/.
- The application has not been run or tested in .venv/.
- No test infrastructure, coverage configuration, or source migration has been implemented.

## 6. Pending work

### Immediate task: WP1.1 — read-only dependency-lock review

Do not modify files during WP1.1. Inspect the generated locks and report:

- Exact PyTorch version and how its dependency metadata brings CUDA packages.
- CUDA Toolkit, Triton, and NVIDIA runtime dependencies and their likely installation implications.
- CPU-only versus GPU-capable PyTorch implications for this project and deployment environments.
- Lock integrity: exact pins are present, but the generated lock files do not contain package hashes. Decide whether that is acceptable for the current policy or should be addressed in a later authorized change.
- Python 3.14 and Linux x86-64 compatibility evidence, while distinguishing resolution/wheel availability from actual package installation and runtime behavior.

Observed lock contents to guide that review:

- torch==2.15.0.
- triton==3.9.0.
- CUDA packages include cuda-bindings==13.4.4, cuda-pathfinder==1.8.5, and cuda-toolkit==13.2.2.
- NVIDIA packages include nvidia-cublas, nvidia-cuda-cccl, nvidia-cuda-cupti, nvidia-cuda-nvrtc, nvidia-cuda-runtime, nvidia-cudnn-cu13, nvidia-cufft, nvidia-cufile, nvidia-cusolver, nvidia-cusparse, nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvjitlink, nvidia-nvshmem-cu13, and nvidia-nvtx.
- faiss-cpu==1.15.1 is the selected FAISS package.
- Lock files list versions and dependency provenance but no hashes or wheel URLs. They are resolution outputs, not evidence that packages are installed or that application behavior works.
- The source calls torch.cuda.is_available() and selects cuda or cpu accordingly in index construction and retrieval. Actual GPU behavior has not been tested.

Treat statements about runtime behavior, CPU fallback with the selected Torch wheel, CUDA driver requirements, and deployment suitability as unresolved until supported by read-only package metadata or a later authorized install test. Do not adjust versions, indexes, manifests, or locks during WP1.1.

### Later work, only after review and authorization

- Address any approved lock-file changes from WP1.1.
- Plan tests and coverage configuration for the 80% production-package line coverage target.
- Begin incremental migration to the target architecture only after an explicit, bounded work package. Do not begin WP2 or later-phase work during WP1/WP1.1.

## 7. Risks

1. **CUDA dependency size and CPU hosts:** the lock includes CUDA/NVIDIA packages even though the project also selects CPU when CUDA is unavailable. Installation footprint and actual fallback behavior remain untested.
2. **Resolution is not runtime verification:** pip-tools resolution succeeded, but no production packages were installed. ABI loading, model initialization, model downloads, and application behavior are unknown.
3. **Lock integrity:** version pins are present, but hashes are absent. This is a lock-policy question for WP1.1; do not silently rewrite the lock.
4. **Existing artifact preservation:** tracked FAISS index and mapping can be overwritten by the current index builder. Tracked PDF and text fixtures must also remain unchanged.
5. **Untracked user files:** .idea/ contains IntelliJ project state. Preserve it and do not stage it as part of routine changes.
6. **Old environment:** the existing venv/ is inconsistent and is not the new target environment. Do not use or modify it.
7. **Shell path leakage:** inherited ROS PYTHONPATH can make venv tools see packages outside .venv/; clear it for venv commands when present.
8. **Incremental migration:** changing imports and package boundaries too broadly can break existing Streamlit entry points, Ollama/Groq modes, or command-line ingestion. Keep each migration step small and reviewable.

## 8. Next exact actions

1. Start WP1.1 as a read-only review. Record current Git status first.
2. Inspect requirements.txt, requirements-dev.txt, and the source calls to torch.cuda.is_available(); do not alter manifests or lock files.
3. Establish package metadata and platform-wheel facts for Torch 2.15.0, Triton 3.9.0, CUDA Toolkit, NVIDIA dependencies, and FAISS. Clearly separate package-index/resolution evidence from an installation or runtime test.
4. Report whether the current lock is suitable for CPU-only and GPU-capable environments, and the consequences of the missing hashes.
5. Stop and present findings for review. Do not install production packages, run the app, create tests, migrate source, stage, or commit.
6. Wait for a separately authorized work package before making any changes or beginning WP2.

## 9. Instructions for the next engineering assistant

- Work on one bounded task at a time and audit before implementation.
- Treat approved product direction as a target architecture, not permission to implement all of it now.
- For WP1.1, remain read-only. Do not change manifests, locks, environment files, source, or data artifacts.
- Preserve Git history. Do not stage or commit without explicit review.
- Preserve .idea/, the old venv/, all three PDFs and extracted text files, and the tracked FAISS index and mapping.
- Do not claim tests, package installations, or application behavior were verified unless those actions actually occurred.
- Do not proceed to testing infrastructure, package migration, FastAPI, Next.js, PostgreSQL/pgvector, object storage, authentication, workers, or other later phases before they are explicitly authorized.
- Prefer maintainable, reviewable changes over adding features.

