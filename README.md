# Sarthink

**Sarthink** is an experimental system for visualizing and exploring your personal digital history as a unified social graph. It bridges the gap between fragmented social media archives (Twitter, Reddit, Discord, etc.) by reconstructing conversation threads and mapping your identity across platforms.

<video controls src="graph-thing.mp4" title="Title"></video>
![Spherical Nodes](image.png)


## Currently at Stage 2 (Memory Graph)

> [!WARNING]
> **EXPERIMENTAL** This is an experimental project and may not be suitable for production use. I do not guarantee if it will work for you. There can be a lot of difference between our data exports and the way they are processed.

## Project Architecture

The project is structured into three main layers:

1.  **Ingestion & Context** (`context/`): Tools to bridge the "missing link" in data exports. While social media archives often only include your own messages, these scripts fetch the surrounding conversation context (replies, parent posts) to reconstruct meaningful threads.
2.  **Parsing & ETL** (`parsers/`): A suite of Python scripts that normalize raw exports and fetched context into a structured SQLite database and JSONL logs.
3.  **Visualization** (`sarthink_graph.html`): A high-performance 3D memory graph rendered via Three.js, allowing you to fly through your digital clusters.

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Social media data exports (Twitter Takeout, Reddit Export, etc.)

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
pip install requests asyncpraw ijson pandas
```

Configure your credentials by copying the example environment file:
```bash
cp .env.example .env
```
Fill in your API keys for Twitter (SocialData) and Reddit (OAuth) in the `.env` file.

### 3. Identity Mapping
To group your nodes correctly across platforms, define your handles in `config/identity_map.json`. You can use the provided example as a template:
```bash
cp config/identity_map.json.example config/identity_map.json
```

### 4. Data Archive Placement
Sarthink dynamically searches your repository for data, but relies on a standard `archive/` folder at the root of the project to locate your raw data exports safely (since it is heavily git-ignored). Organize your exports exactly like this:
```text
sarthink/
├── archive/
│   ├── reddit-export/          # Folder containing your Reddit posts.csv, comments.csv, etc.
│   ├── discord-export/         # Folder containing your Discord JSON exports
│   ├── any_meta_export.zip     # Raw Meta GDPR zips (make sure 'facebook' or 'instagram' is in the filename)
│   └── tweets.js               # Twitter JS files (these are found recursively anywhere in the repo)
```

## Workflow

### A. Context Fetching
Before parsing, you may need to fetch the conversation context that isn't included in your raw exports:
- **Twitter**: Run `context/twitter/fetch_context.py` to crawl reply chains.
- **Reddit**: Run `context/reddit/main.py` to fetch missing parent posts and replies.

### B. Parsing Data
Run the platform-specific parsers to populate the database:
```bash
python3 parsers/twitter_parser.py
python3 parsers/reddit_parser.py
# ... etc
```

### C. Graph Generation
Compute the 3D layout and export the data for the web UI:
```bash
python3 parsers/compute_layout.py
python3 parsers/export_cosmograph.py
```

### D. Visualizing
The graph is a single-file web application. Due to CORS restrictions when loading CSV files, you must run it through a local web server:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000/sarthink_graph.html`.

