# VectorDB

> A vector database built from scratch in C++ — HNSW, KD-Tree, Brute Force, and a full RAG pipeline powered by a local LLM.

---

## Tech Stack

<div align="center">

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Canvas-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES2022-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-FFFFFF?style=for-the-badge&logo=ollama&logoColor=black)
![REST](https://img.shields.io/badge/REST-API-6366F1?style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-22D3EE?style=for-the-badge)

</div>

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form in real time |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works
Your Text
│
▼
Ollama (nomic-embed-text) ← converts text to a 768-dimensional vector
│
▼
HNSW Index (C++) ← indexes the vector in a multilayer graph
│
▼
Semantic Search ← finds nearest neighbors in vector space
│
▼
Ollama (llama3.2) ← reads retrieved chunks, generates an answer
│
▼
Answer

text


**HNSW** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving **O(log N)** complexity instead of O(N) for brute force.

---

## Prerequisites

Three things to install:

- **MSYS2** — gives you the `g++` compiler on Windows
- **Git**
- **Ollama** — runs the local AI models

---

## Step-by-Step Setup (Windows)

### Step 1 — Install MSYS2 (C++ Compiler)

1. Go to [https://www.msys2.org](https://www.msys2.org) and download the installer
2. Run it, keep the default path (`C:\msys64`)
3. Open **MSYS2 UCRT64** from the Start Menu (the orange icon)
4. Run:

```bash
pacman -Syu
Close and reopen the terminal if it asks you to restart.

Bash

pacman -S mingw-w64-ucrt-x86_64-gcc
Add g++ to your Windows PATH:

Press Win + R, type sysdm.cpl, press Enter
Click Advanced → Environment Variables
Under System variables, find Path, click Edit
Click New and add: C:\msys64\ucrt64\bin
Click OK on all windows
Open a new PowerShell and verify:

Bash

g++ --version
# Expected: g++ (GCC) 15.x.x
Step 2 — Install Git
Go to https://git-scm.com/download/win, run the installer with default settings.

Bash

git --version
Step 3 — Install Ollama
Go to https://ollama.com and click Download for Windows
Run the installer — Ollama starts automatically in the system tray
Open PowerShell and pull the two required models:
Bash

ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — language model
Verify both are ready:
Bash

ollama list
Minimum specs: 8 GB RAM recommended. Both models use ~3 GB total.

Step 4 — Clone the Repository
Bash

git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
Step 5 — Compile the C++ Server
Bash

g++ -std=c++17 -O2 main.cpp -o db -lws2_32
This produces db.exe. It takes about 10–20 seconds.

Error	Fix
g++: command not found	MSYS2 not in PATH — redo Step 1 point 5
undefined reference to WSA...	Add -lws2_32 to the compile command
Step 6 — Run Everything
Terminal 1 — Start Ollama (skip if already running in system tray):

Bash

ollama serve
Terminal 2 — Start the VectorDB server:

Bash

./db
You should see:

text

  ╔══════════════════════════════════════════╗
  ║         MY OWN AI — VectorDB Engine      ║
  ╠══════════════════════════════════════════╣
  ║  Server:    http://localhost:8080         ║
  ║  Vectors:    20 demo items (16D)          ║
  ║  Algorithms: HNSW + KD-Tree + BruteForce ║
  ║  Ollama:    ONLINE ✓                      ║
  ║    embed:   nomic-embed-text              ║
  ║    gen:     llama3.2                      ║
  ╚══════════════════════════════════════════╝
Open your browser and go to:

text

http://localhost:8080
Using the Application
Tab 1 — Search (Demo Vectors)
Type any concept: binary tree, sushi, basketball, calculus

Choose your algorithm: HNSW, KD-Tree, or Brute Force
Choose a distance metric: Cosine, Euclidean, or Manhattan
Click ⚡ SEARCH — results appear with distances, the matching point glows on the scatter plot
Click ▶ COMPARE ALL ALGOS to benchmark all 3 algorithms side by side
The scatter plot projects all 20 vectors to 2D using PCA. The 4 semantic categories (CS, Math, Food, Sports) form distinct visible clusters — this is what semantic similarity looks like visually.

Tab 2 — Documents (Real Embeddings)
Enter a title (e.g., Operating Systems Notes)
Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
Click ⚡ EMBED & INSERT
Long documents are automatically split into overlapping 250-word chunks. Each chunk gets its own 768D embedding stored in a separate HNSW index.

Tab 3 — Ask AI (RAG Pipeline)
Insert documents first, then type a question and click 🤖 ASK AI.

What happens behind the scenes:

text

1. Your question  →  embedded with nomic-embed-text  →  768D vector
2. HNSW search    →  finds 3 most semantically similar chunks
3. Retrieved chunks  →  sent as context to llama3.2
4. llama3.2       →  generates an answer grounded in your documents
Click the context chips below each answer to see exactly which chunks the AI used.

REST API Reference
Demo Vector Endpoints
Method	Endpoint	Description
GET	/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw	K-NN search
POST	/insert	Insert a demo vector
DELETE	/delete/:id	Delete by ID
GET	/items	List all demo vectors
GET	/benchmark?v=...&k=5&metric=cosine	Compare all 3 algorithms
GET	/hnsw-info	HNSW graph structure and layer stats
GET	/stats	Database statistics
Document & RAG Endpoints
Method	Endpoint	Body	Description
POST	/doc/insert	{"title":"...","text":"..."}	Embed and store document
GET	/doc/list	—	List all stored documents
DELETE	/doc/delete/:id	—	Delete document chunk
POST	/doc/ask	{"question":"...","k":3}	RAG: retrieve + generate
GET	/status	—	Ollama status and model info
Examples
Bash

# Search via curl
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

# Ask a question via curl (PowerShell)
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
Project Structure
text

VectorDB/
├── main.cpp      ← C++ backend: HNSW, KD-Tree, BruteForce, REST API, RAG pipeline
├── httplib.h     ← Single-header HTTP server library (cpp-httplib)
├── index.html    ← Frontend: PCA scatter plot, chat UI, benchmark visualizer
└── README.md     ← This file
Architecture (main.cpp)
text

BruteForce     O(N·d)     Exact, baseline — compares query against every vector
KDTree         O(log N)   Exact, binary space partitioning along cycling axes
HNSW           O(log N)   Approximate, multilayer small-world graph

VectorDB       Unified interface over all 3 indexes (16D demo vectors)
DocumentDB     HNSW-only index for real Ollama embeddings (768D)
OllamaClient   HTTP client → /api/embeddings + /api/generate
Algorithm Deep Dive
HNSW — Hierarchical Navigable Small World
Nodes are inserted into a multilayer graph. Each node is randomly assigned a maximum layer. Layer 0 contains all nodes with dense connections; higher layers have exponentially fewer nodes with longer-range connections.

Insert: Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from the assigned max down to 0, run a beam search (ef_construction = 200) and connect to the M nearest neighbors bidirectionally.

Search: Same greedy descent from the top layer. At layer 0, expand to ef nearest candidates using a min-heap priority queue.

Why it's fast: The upper layers act like a highway — you quickly arrive in the right neighborhood, then zoom in precisely at layer 0.

KD-Tree — K-Dimensional Tree
Binary space partitioning. Each internal node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree cannot beat the current best — the "ball within hyperslab" check.

Weakness: Degrades with high dimensions (curse of dimensionality). Works well at ≤20D. Approaches brute force at 768D.

Why HNSW Wins at High Dimensions
KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all volume concentrates near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach does not have this problem.

Common Issues
Problem	Fix
Ollama shows OFFLINE in the header	Run ollama serve in a terminal
Embedding takes a very long time	Ollama is downloading the model on first use — wait 2 minutes
g++: command not found	Add C:\msys64\ucrt64\bin to Windows PATH
Port 8080 already in use	netstat -ano | findstr 8080 then taskkill /PID <pid> /F
LLM answers are slow	Normal — llama3.2 takes 10–30s on a laptop CPU. See below for a faster model
Use a Smaller, Faster LLM
If llama3.2 is too slow on your machine:

Bash

ollama pull llama3.2:1b
Then change one line in main.cpp:

C++

std::string genModel = "llama3.2:1b";   // was "llama3.2"
Recompile and restart.
