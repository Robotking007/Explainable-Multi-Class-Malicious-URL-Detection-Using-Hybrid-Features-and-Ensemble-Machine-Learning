# Explainable AI Cybersecurity Threat Intelligence Platform
## A Graph Neural Network-Based Framework for Explainable Zero-Day Malware and Malicious-URL Behavior Analysis

### EDAI College Capstone Project — Production SOC Frontend

A responsive, high-performance, dark SOC-themed threat intelligence dashboard built with React 19, Vite, Tailwind CSS, Recharts, React Flow, and Framer Motion. Connects directly to FastAPI backend services served via Uvicorn.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
cd FRONTED
npm install
```

### 2. Configure Backend Connection
Copy the sample environment file:
```bash
cp .env.example .env
```
Default configuration points to local FastAPI on port 8000:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_APP_TITLE=CyberX AI Threat Intelligence
```
*(You can also modify the backend address at runtime in the **Settings & Status** tab.)*

### 3. Launch Development Server
```bash
npm run dev
```
Open your browser at `http://localhost:5173`.

### 4. Build for Production
```bash
npm run build
npm run preview
```

---

## 🛡️ Architecture & Feature Suite

| Module | Route | Capabilities |
| :--- | :--- | :--- |
| **Overview Dashboard** | `/` | Real risk distribution pie chart, classification bar chart, live telemetry, GNN module availability status. Zero fabricated metrics. |
| **Threat Analyzer** | `/threat-analyzer` | Unified analysis workspace. **Mode A**: Malicious URL classification with sample presets & request cancellation. **Mode B**: Malware analysis with 3 project modules. |
| **URL Intelligence** | `/url-intelligence` | Domain age, WHOIS registrar data, IP resolution, Autonomous System (ASN) prefix routing, DNS records. |
| **Malware Analysis** | `/malware-analysis` | Deep-dive workbench for **CIC-MalMem-2022** (memory dumps), **BODMAS & EMBER-2018** (PE binaries), and **MalAPI2019** (syscall sequences). |
| **Infrastructure Graph** | `/infrastructure-graph` | Interactive React Flow spatial graph mapping URL → Domain → IP → ASN relations. Includes labeled demo mode if URLInfraGNN is offline. |
| **Dynamic Analysis** | `/dynamic-analysis` | 5-stage pipeline stepper: Static Parser → Confidence Gating → Headless Browser Sandbox → DOM & JS Profiler → Multi-Modal Verdict. |
| **Explainable AI** | `/explainable-ai` | SHAP feature attribution waterfall/bar chart, positive vs. negative weights, neural gradient layer sensitivities, transparent reasoning summary. |
| **Scan History** | `/scan-history` | Search, filter by severity/type, sortable telemetry table, pagination, detail drawer, and CSV export. |
| **AI Security Assistant** | `/security-assistant` | Full-page SOC copilot + persistent floating widget with suggested cybersecurity questions and backend LLM integration. |
| **Settings & Status** | `/settings` | Live backend health checker, latency (RTT) measurement, dynamic base URL reconfiguration, and model checkpoint inspector. |

---

## 🔌 FastAPI Backend REST Contracts

The frontend queries the following REST endpoints on your FastAPI backend (`main.py`):

### 1. Health Probe
- **`GET /health`** or **`GET /api/health`**
- **Response**: `{"status": "ok", "version": "1.0.0", "gnn_models_loaded": true}`

### 2. URL Threat Analysis
- **`POST /api/analyze-url`**
- **Request Body**:
  ```json
  { "url": "http://suspicious-domain.com/login.php" }
  ```
- **Response Format**:
  ```json
  {
    "scan_id": "scan-abc123",
    "url": "http://suspicious-domain.com/login.php",
    "classification": "Phishing",
    "risk_level": "Malicious",
    "confidence": 0.942,
    "features": {
      "url_length": 45,
      "has_ip": false,
      "entropy": 4.82
    },
    "explanation": "High token entropy combined with recent domain registration."
  }
  ```

### 3. Malware Analysis (Multipart Form)
- **`POST /api/analyze-malware`**
- **Form Data**:
  - `file`: Binary file upload
  - `module_id`: `'cic-malmem-2022'` | `'bodmas-ember'` | `'malapi-2019'`
- **Response Format**:
  ```json
  {
    "scan_id": "mal-xyz789",
    "filename": "payload.raw",
    "malware_family": "Ransomware.WannaCry",
    "risk_level": "Critical",
    "confidence": 0.978,
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
  ```

### 4. URL & Host Intelligence
- **`POST /api/intelligence/url`**
- **Request**: `{ "url": "https://example.com" }`
- **Response**: `{ "domain": "example.com", "domain_age": "450 days", "ip": "93.184.216.34", "asn": "AS15133", "whois": { ... } }`

### 5. Infrastructure Graph Topology
- **`POST /api/graph/infrastructure`**
- **Request**: `{ "target": "https://example.com" }`
- **Response**: `{ "nodes": [...], "edges": [...] }`

### 6. Dynamic URL Sandbox Inspection
- **`POST /api/dynamic-analysis/start`**
- **Request**: `{ "url": "https://example.com" }`

### 7. Explainable AI (XAI) Attributions
- **`POST /api/xai/url`**
- **Request**: `{ "url": "https://example.com", "model": "URLInfraGNN" }`

### 8. AI Chatbot
- **`POST /api/chat`**
- **Request**:
  ```json
  {
    "message": "Why was this URL classified as phishing?",
    "history": [],
    "context": { "target": "https://example.com", "risk": "Malicious" }
  }
  ```
- **Response**: `{ "response": "The model flagged this URL because..." }`

---

## 🔒 Security & SOC Safety
- **Safe Handling of Untrusted Files**: Uploaded malware samples are never executed in the client browser.
- **Defensive Sandbox Notice**: Dynamic inspection is contained within backend headless environments.
- **Interpretability Disclaimer**: SHAP feature importance reflects mathematical gradient sensitivity rather than physical causation.
