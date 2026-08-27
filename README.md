# FinTwin — Agentic AI for Banking Customer Engagement

> Built for the SBI Banking AI Hackathon · June 2026

FinTwin is a real-time, event-driven AI system that detects significant changes in a customer's financial life, reasons about them using a multi-step agentic pipeline with genuine tool-calling, and automatically delivers personalized outreach — all without human intervention after the system is started.

---

## What it actually does

1. **Detects life events in real time** — a continuous transaction simulator generates new activity for 500 synthetic customers. A rules-based detector watches each transaction as it lands, identifying meaningful financial events: salary jumps, savings milestones, and new financial commitments.

2. **Reasons agentically, not with a single prompt** — when an event is detected, an LLM agent (Llama 3.3 70B via Groq) runs a genuine 4-step pipeline:
   - **INVESTIGATE** — the agent calls real tools to check the customer's situation before deciding anything: `check_existing_products`, `calculate_affordability`, `get_risk_flags`. It chooses which tools to call based on context, not a fixed sequence.
   - **DECIDE** — grounded in what it found, the agent selects appropriate products and assigns an outreach priority.
   - **DRAFT** — writes a personalized outreach message referencing the customer's real numbers, not a template.
   - **VALIDATE** — an independent compliance pass reviews the draft for banking-inappropriate language (guaranteed returns, false urgency, unsupported claims) and auto-revises if needed.

3. **Delivers real email** — approved drafts are delivered as branded HTML emails via n8n + Gmail, with a full product recommendation table, action links, and personalized content. Proven working end-to-end.

4. **Shows every reasoning step** — the dashboard renders the full agent trace (every tool call, every intermediate result, the compliance check outcome) so the reasoning process is transparent and auditable, not a black box.

---

## Architecture

```
[Transaction Simulator] → [Rules-based Detector] → [asyncio.Queue]
                                                           ↓
                                              [Agent Worker Loop]
                                                    ↓
                                         [LLM Agent Pipeline]
                                         1. INVESTIGATE (tool calls)
                                         2. DECIDE (recommendation)
                                         3. DRAFT (outreach message)
                                         4. VALIDATE (compliance check)
                                                    ↓
                              [WebSocket broadcast] → [Browser UI]
                                                    ↓
                                         [n8n webhook] → [Gmail]
                                                    ↓
                                            [SQLite database]
```

**Tech stack (honest):**
- Python 3.9, FastAPI, Uvicorn, asyncio
- Groq API (Llama 3.3 70B) for LLM reasoning
- SQLite (WAL mode) for persistent storage
- sentence-transformers + local embeddings for RAG knowledge base
- n8n + Gmail for real email delivery
- Vanilla JS + WebSockets for live UI updates
- GitHub Actions not yet set up (planned)

---

## Key files

| File | What it does |
|---|---|
| `live_feed.py` | Continuous transaction simulator with realistic monthly cycles and life-event injection |
| `agent_engine.py` | Core agentic pipeline: tool definitions, investigate/decide/draft/validate loop, full trace logging |
| `detect_events.py` | Rules-based life-event detector, shared by batch and live paths |
| `server.py` | FastAPI backend: WebSocket push, feed controls, REST API, async agent worker queue |
| `db.py` | SQLite data layer: customers, events, agent runs, trace steps, email delivery log |
| `rag_engine.py` | Local RAG: sentence-transformer embeddings, semantic search over product knowledge base |
| `eval_scenarios.py` | 20+ hand-authored test scenarios with expected outcomes across 3 evaluation dimensions |
| `run_eval.py` | Evaluation harness: runs scenarios, scores pass/fail per dimension, logs to database |
| `static/index.html` | Live dashboard: customer queue, agent trace timeline, speed controls, outreach editor |
| `knowledge_base/` | Synthetic SBI-style product documents (clearly labeled illustrative, not official) |

---

## Evaluation results

Tested against a hand-authored suite of scenarios covering three dimensions:

| Dimension | Pass rate | What it measures |
|---|---|---|
| Tool usage | 100% (14/14) | Did the agent call the right investigation tools before deciding? |
| Product fit | 93% (13/14) | Did final recommendations match the customer's real financial situation? |
| Compliance | 93% (13/14) | Did the validation step catch and revise inappropriate banking language? |

One known gap identified: the compliance validator missed an exaggerated-promises case where the draft used implied-certainty language for a market-linked product. Root cause identified and fix in progress (see `run_eval.py` and `eval_scenarios.py`).

---

## RAG knowledge base

A local retrieval-augmented generation layer (`rag_engine.py`) embeds 10 synthetic SBI-style product documents using `sentence-transformers/all-MiniLM-L6-v2` (runs fully offline, no API key needed) and exposes a `search_product_policy(query)` function for semantic retrieval. Integration with the live agent pipeline is in progress.

Sample retrieval result for query *"personal loan eligibility for a 22 year old"*:
- Top result: SBI Personal Loan eligibility criteria (similarity: 0.63) — correctly surfaces minimum age 21, DTI ratio limit, credit score requirement.

---

## Running locally

```bash
# Clone and set up
git clone https://github.com/raikwarxdev/FinTwin.git
cd FinTwin
pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# Generate synthetic data (first time only)
python3 generate_data.py
python3 detect_events.py

# Start the server
python3 -u server.py

# Open http://localhost:8000
# Click "Start Feed" to watch live event detection and agent reasoning
```

---

## What's genuinely in this repo vs. what's planned

**Working and verified:**
- Live transaction simulation and event detection
- Full agentic pipeline with real tool-calling (not prompt chaining)
- Real email delivery via n8n integration
- SQLite data layer with normalized schema
- Local RAG infrastructure (retrieval proven, agent integration in progress)
- Evaluation framework with real measured pass rates

**In progress / planned:**
- RAG integration with live agent tool-calling
- Compliance validator fix for the identified gap
- Conversation memory per customer (so agent references prior outreach history)
- Dashboard authentication

---

## Honest notes

- Customer data is entirely synthetic — generated by `generate_data.py`, not real SBI data
- Product documents in `knowledge_base/` are illustrative descriptions written for this project, not official SBI rates or terms
- Email delivery is routed to test inboxes we control; in production this would route to customers' registered emails
- The system runs as a single FastAPI process — this is appropriate for a prototype at this scale, not a production deployment concern

---

*Built by Dev Kumar Raikwar · github.com/raikwarxdev/FinTwin*
