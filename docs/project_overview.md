# Sarthink: Project Overview & Roadmap

Sarthink is a high-performance personal memory layer designed to visualize, search, and eventually "reason" across your entire digital conversational history (2015–Present).

## 1. Initial Requirements
*   **Omni-Platform Ingestion:** Support for Twitter/X, Reddit, Discord, and Meta (Instagram/Facebook).
*   **Scale & Performance:** Handle 300,000+ messages and 80,000+ nodes at 60fps in the browser.
*   **Identity Resolution:** A curated mapping system to collapse disparate usernames across platforms into a single "Master Persona (me)" entity.
*   **Contextual Integrity:** Fetching missing parent threads (via Twitter/Reddit APIs) to ensure the graph isn't just "you talking into a void."
*   **Cinematic Visualization:** A pre-computed 3D layout that highlights "Mega-clusters" and allows for multi-hop neighbor isolation.

---

## 2. Project Structure

### `/parsers` (Ingestion)
*   `database.py`: The core `SarthinkMemoryLayer` (SQLite + JSONL dual-write).
*   `*_parser.py`: Platform-specific normalization logic.
*   `export_cosmograph.py`: Collapses messages into relational vectors (User ↔ Thread).
*   `compute_layout.py`: The Python-based geometric engine that pre-calculates 3D coordinates.

### `/context` (Enrichment)
*   Scripts to crawl external APIs for missing message parents, filling in the gaps that standard platform archives leave out.

### `/processed_data` (Storage)
*   `sarthink_memory.db`: Relational skeleton (Who/Where/When).
*   `*_logs.jsonl`: The Content Logs (The "Flesh" for LLM processing).
*   `cosmograph_*.csv`: The baked data consumed by the 3D renderer.

### `sarthink_graph.html` (The "Face")
*   A pure Three.js/WebGL renderer. Zero-physics, GPU-bound, supporting interactive cluster dragging and cinematic recording.

---

## 3. Roadmap: Moving Forward

### Stage 2: Semantic Intelligence (The "Brain")
The next phase transitions the graph from a structural map to a queryable mind.

#### **The Chunking Strategy**
Unlike standard RAG (which chunks by word count), Sarthink uses **Temporal-Relational Chunking**:
*   **Window:** Messages within the same thread.
*   **Density:** At least 3 messages (to/from).
*   **Time-box:** Interactions occurring within a 16-hour window.
*   **Goal:** These chunks form "Conversation Blocks" which are then embedded as single vectors to capture the *vibe* and *topic* of a specific interaction.

#### **Neural Search**
*   **Local Vector Store:** Indexing the blocks using a local GPU-bound embedding model (e.g., `BAAI/bge-large-en-v1.5`).
*   **Semantic Highlighting:** Searching for "photography" should light up clusters even if the word "photography" isn't in the label, based on conversation content.

### Stage 3: The Second Brain
*   **RAG Summaries:** Clicking a node triggers an LLM to "read" the JSONL logs for that connection and summarize your history with that person.
*   **Persona Training:** Using the standardized JSONL outputs to fine-tune an LLM on your specific conversational patterns, tone, and logic.

---

## 4. Why this Structure?
By separating **Skeleton (SQLite)**, **Positions (CSV/Layout)**, and **Content (JSONL)**, we ensure the visualization remains buttery smooth while the AI has structured, high-context data to reason over.
