"""
FinTwin — Agentic Engine (v2)
--------------------------------
This is the actual "agentic AI" layer: instead of one prompt -> one answer,
the LLM is given real tools and decides FOR ITSELF which to call, in what
order, before producing a recommendation. Every decision is logged so the
full reasoning trace can be shown in the UI (this is what makes it
demonstrably agentic, not just a single completion with a nice prompt).

Pipeline per customer:
  1. INVESTIGATE — model calls tools (existing products, affordability,
     risk flags) as it decides it needs them. Not all tools are called for
     every customer; the model chooses based on the situation.
  2. DECIDE — model produces a recommendation + priority, grounded in what
     it found during investigation.
  3. DRAFT — model writes the customer-facing outreach message.
  4. VALIDATE — a second, independent pass checks the draft against basic
     banking-compliance rules (no guaranteed-return language, no false
     promises, no aggressive pressure tactics) and flags/fixes issues.

Output: data/agent_runs.json — a list of full traces (one per customer),
each containing every tool call, its result, and the final output. The
frontend renders this trace directly so judges can see the agent "think."

Setup:
    GROQ_API_KEY in .env (https://console.groq.com/keys, free)

Run:
    python3 agent_engine.py
"""

import json
import os
import sys
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import db

load_dotenv()

# ---------------------------------------------------------------------------
# Multi-key rotation (3 free Groq keys = 300,000 tokens/day combined)
# ---------------------------------------------------------------------------

_GROQ_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY"),  # fallback: single key setup
    ] if k
]

if not _GROQ_KEYS:
    sys.exit(
        "ERROR: No Groq API key found. Add GROQ_API_KEY_1, GROQ_API_KEY_2, "
        "GROQ_API_KEY_3 (or at minimum GROQ_API_KEY) to your .env file."
    )

_current_key_index = [0]

def get_client() -> OpenAI:
    """Return an OpenAI-compatible Groq client using the current active key."""
    key = _GROQ_KEYS[_current_key_index[0] % len(_GROQ_KEYS)]
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def rotate_key(reason: str = "quota"):
    """Rotate to the next available key and log the reason."""
    _current_key_index[0] = (_current_key_index[0] + 1) % len(_GROQ_KEYS)
    print(f"[KeyRotation] Rotated to key index {_current_key_index[0]} (reason: {reason}). "
          f"{len(_GROQ_KEYS)} keys available.")

MODEL_NAME = os.getenv("FINTWIN_MODEL", "openai/gpt-oss-120b")

DATA_DIR = Path("data")
EVENTS_FILE = DATA_DIR / "detected_events.json"
CUSTOMERS_FILE = DATA_DIR / "customers.json"
OUTPUT_FILE = DATA_DIR / "agent_runs.json"

MAX_CUSTOMERS = 15
SECONDS_BETWEEN_CUSTOMERS = 2


# ---------------------------------------------------------------------------
# TOOLS — real functions the agent can choose to call
# ---------------------------------------------------------------------------
# These run locally against the customer/transaction data we already have.
# In a production system these would hit real banking APIs/databases; here
# they read from the same JSON files Day 1 produced, which is an honest
# representation of "the agent queries internal systems before acting."

def check_existing_products(customer: dict) -> dict:
    """Simulates checking the bank's core system for what the customer
    already holds, so the agent doesn't recommend something they have."""
    has_emi = customer.get("has_emi", False)
    emi_amount = customer.get("existing_emi_amount", 0)
    return {
        "has_active_loan": bool(has_emi and emi_amount and emi_amount > 0),
        "existing_emi_amount": emi_amount,
        "note": "Customer already has an active EMI obligation" if (has_emi and emi_amount) else "No active loan on record",
    }


def calculate_affordability(customer: dict) -> dict:
    """Computes a simple affordability ratio to ground loan/investment
    recommendations in the customer's real capacity, not just the event."""
    salary = customer.get("base_salary", 0) or 0
    emi = customer.get("existing_emi_amount", 0) or 0
    savings = customer.get("savings_balance", 0) or 0

    emi_to_income = round((emi / salary) * 100, 1) if salary else None
    months_of_runway = round(savings / salary, 1) if salary else None

    if emi_to_income is not None and emi_to_income > 40:
        risk_band = "high"
    elif emi_to_income is not None and emi_to_income > 20:
        risk_band = "medium"
    else:
        risk_band = "low"

    return {
        "monthly_salary": salary,
        "existing_emi_to_income_pct": emi_to_income,
        "savings_runway_months": months_of_runway,
        "debt_burden_risk_band": risk_band,
    }


def get_risk_flags(customer: dict, event: dict) -> dict:
    """Surfaces simple risk/eligibility flags an RM would want before
    reaching out, e.g. very young/old age for certain products, very
    recent event, low confidence detection."""
    flags = []
    age = customer.get("age")
    if age is not None and age < 21:
        flags.append("Customer is under 21 — restrict loan/credit product offers")
    if age is not None and age > 60:
        flags.append("Customer is 60+ — prioritize savings/insurance over long-tenure loans")
    confidence = event.get("confidence", 1)
    if confidence is not None and confidence < 0.7:
        flags.append("Event detection confidence is low — consider a softer, exploratory outreach tone")
    return {"flags": flags, "flag_count": len(flags)}


from rag_engine import search_product_policy as _search_product_policy

def search_product_policy_tool(query: str) -> dict:
    results = _search_product_policy(query, top_k=2)
    return {
        "results": [
            {
                "product": r["product_name"],
                "section": r["section_name"],
                "content": r["text"],
                "relevance": round(r["similarity_score"], 3)
            }
            for r in results
        ],
        "note": "Retrieved from local SBI product knowledge base (illustrative/synthetic rates)"
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_existing_products",
            "description": "Check what financial products (loans, accounts) the customer already holds, to avoid recommending something redundant.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_affordability",
            "description": "Calculate the customer's debt-to-income ratio and savings runway to assess whether they can realistically afford a new loan/investment product.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_flags",
            "description": "Get any eligibility or risk flags (age restrictions, low-confidence detection) relevant to this customer and event before recommending outreach.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product_policy",
            "description": "Search the SBI product knowledge base for eligibility criteria, interest rates, tenure limits, fees, and age restrictions for a specific product or customer situation. Call this before recommending any specific product to ground your recommendation in actual policy, not just general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you need to know, e.g. 'personal loan eligibility for a 28 year old with existing EMI' or 'fixed deposit interest rates for senior citizen'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "check_existing_products": check_existing_products,
    "calculate_affordability": calculate_affordability,
    "get_risk_flags": get_risk_flags,
    "search_product_policy": search_product_policy_tool,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run Day 1 scripts first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_customers_by_id(customers: list[dict]) -> dict:
    return {c["customer_id"]: c for c in customers}


def pick_subset(events: list[dict], max_count: int) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("event_type", "unknown"), []).append(e)
    types = list(by_type.keys())
    subset = []
    i = 0
    while len(subset) < max_count and any(by_type[t] for t in types):
        t = types[i % len(types)]
        if by_type[t]:
            subset.append(by_type[t].pop(0))
        i += 1
    return subset


def clean_llm_json(text: str) -> str:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    elif "<think>" in text:
        text = text.split("<think>", 1)[1]
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        else:
            first_brace = text.find("{")
            if first_brace != -1:
                text = text[first_brace:].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    # Extract JSON object even if LLM adds preamble text before it
    if not text.strip().startswith("{"):
        import re as _re
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            text = m.group(0)
        text = text.strip("`").strip()
    return text


# ---------------------------------------------------------------------------
# STEP 1+2: INVESTIGATE + DECIDE (agentic loop with tool calling)
# ---------------------------------------------------------------------------

INVESTIGATE_SYSTEM_PROMPT = """You are an AI agent for SBI relationship managers. A customer life \
event has been detected. Before recommending anything, investigate the customer's real financial \
situation using the tools available to you. Call whichever tools are relevant — you don't need to \
call all of them if the situation is simple, but for anything involving a loan or investment \
recommendation, check affordability and existing products first. You MUST call search_product_policy \
for every candidate product mentioned in the user prompt to verify its eligibility criteria, rates, \
and restrictions before deciding whether to recommend or drop it — this grounds your recommendation \
in real policy rather than general knowledge.

Banking Guidelines:
- Do NOT recommend any credit or loan products if the customer has severe debt burden (existing DTI > 50%), is underage (under 21 for loans), or fails eligibility criteria.
- Do NOT recommend volatile investment products (like SIP or mutual funds) for senior citizens (60+) or customers with missed EMIs or zero savings runway.
- You SHOULD still recommend zero-risk, zero-balance account upgrades (like SBI Salary Account Upgrade) for salaried customers even if they have high debt or low savings, as it does not add to their debt burden and provides transaction benefits.

Priority Guidelines:
- "low": Use this if event detection confidence is low (< 0.7), if the customer has severe debt burden (existing DTI > 50%), if there are major eligibility/risk flags (e.g. customer is under 21, or too close to retirement), or if you reject/drop the main loan or investment products.
- "medium": Standard cases with clear product fit, reasonable confidence, and acceptable debt burden.
- "high": High-confidence, strong-affordability cases for major events with no eligibility issues.

Once you have enough information, respond with your final decision as JSON (no markdown fences) in this exact shape:
{
  "reasoning": "2-3 sentences explaining the recommendation, grounded in what you found from your tool calls — reference actual numbers",
  "priority": "high | medium | low",
  "priority_reason": "one short sentence explaining the priority choice",
  "final_products": ["product names you actually recommend — you may DROP a candidate product if your investigation shows it's a bad fit"]
}"""


def build_investigate_user_prompt(event: dict, customer: dict) -> str:
    products_list = "\n".join(
        f"- {p['name']} ({p['type']})" for p in event.get("recommended_products", [])
    )
    return f"""CUSTOMER PROFILE:
- Name: {customer.get('name')}
- Age: {customer.get('age')}
- City: {customer.get('city')}
- Base salary: ₹{customer.get('base_salary')}
- Savings balance: ₹{customer.get('savings_balance')}

DETECTED EVENT:
- Type: {event.get('event_label')}
- Signal: {event.get('signal')}
- Confidence: {event.get('confidence')}

CANDIDATE PRODUCTS (from rules engine, not yet verified against customer's real capacity):
{products_list}

Investigate using your tools, then give your final decision as JSON."""


def run_investigate_decide(event: dict, customer: dict, trace: list) -> dict:
    messages = [
        {"role": "system", "content": INVESTIGATE_SYSTEM_PROMPT},
        {"role": "user", "content": build_investigate_user_prompt(event, customer)},
    ]

    max_tool_rounds = 6
    for _ in range(max_tool_rounds):
        for _attempt in range(len(_GROQ_KEYS) + 1):
            try:
                response = get_client().chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=2048,
                )
                break  # success
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                    if _attempt < len(_GROQ_KEYS):
                        rotate_key("429")
                        time.sleep(2)
                        continue
                raise  # non-quota error, re-raise immediately
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn_name == "get_risk_flags":
                    result = fn(customer, event)
                elif fn_name == "search_product_policy":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = fn(**args)
                    except Exception as e:
                        result = {"error": f"failed to call search_product_policy: {e}"}
                else:
                    result = fn(customer) if fn else {"error": f"unknown tool {fn_name}"}

                trace.append({
                    "step": "tool_call",
                    "tool": fn_name,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result),
                })
            continue

        # No more tool calls — model is ready with a final answer
        text = msg.content or ""
        text = clean_llm_json(text)
        try:
            decision = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[DEBUG ERROR] JSON Decode Error: {e} | Raw text: {text!r}", file=sys.stderr)
            decision = {
                "reasoning": text or "Model did not return valid JSON.",
                "priority": "medium",
                "priority_reason": "Default — could not parse model output",
                "final_products": [p["name"] for p in event.get("recommended_products", [])],
            }
        trace.append({"step": "decision", "result": decision})
        return decision

    # Fallback if the model loops on tool calls without converging
    fallback = {
        "reasoning": "Investigation incomplete after maximum tool-call rounds.",
        "priority": "medium",
        "priority_reason": "Fallback — agent did not converge",
        "final_products": [p["name"] for p in event.get("recommended_products", [])],
    }
    trace.append({"step": "decision", "result": fallback})
    return fallback


# ---------------------------------------------------------------------------
# STEP 3: DRAFT outreach message
# ---------------------------------------------------------------------------

def run_draft(event: dict, customer: dict, decision: dict, trace: list) -> dict:
    prompt = f"""Write a short, warm SMS/email a relationship manager can send DIRECTLY to this \
customer. Reference their real situation naturally, don't sound like a generic bank promo.

Customer: {customer.get('name')}
Event: {event.get('event_label')} — {event.get('signal')}
Recommendation reasoning: {decision.get('reasoning')}
Products to mention: {', '.join(decision.get('final_products', []))}

Respond ONLY with JSON, no markdown fences:
{{
  "channel": "sms | email",
  "subject": "only if channel is email, else empty string",
  "body": "the message, max 400 characters, use the customer's real name, no placeholders"
}}"""

    for _attempt in range(len(_GROQ_KEYS) + 1):
        try:
            response = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            break  # success
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                if _attempt < len(_GROQ_KEYS):
                    rotate_key("429")
                    time.sleep(2)
                    continue
            raise  # non-quota error, re-raise immediately
    text = response.choices[0].message.content or ""
    text = clean_llm_json(text)
    try:
        draft = json.loads(text)
        if not isinstance(draft, dict):
            draft = {"channel": "sms", "subject": "", "body": str(draft)[:400]}
    except json.JSONDecodeError:
        draft = {"channel": "sms", "subject": "", "body": text[:400]}

    # Normalize keys to ensure 'body' is populated
    if "body" not in draft or not draft.get("body"):
        for key in ["message", "text", "email_body", "content", "outreach"]:
            if key in draft and draft[key]:
                draft["body"] = draft[key]
                break

    trace.append({"step": "draft", "result": draft})
    return draft


# ---------------------------------------------------------------------------
# STEP 4: VALIDATE — independent compliance check pass
# ---------------------------------------------------------------------------

VALIDATE_PROMPT = """You are a compliance reviewer for an Indian bank's customer outreach messages. \
Review the message below for compliance violations and flag/revise them if present.

Compliance Rules:
1. NO Guaranteed Returns or Assured Growth: Do not allow phrases like "guaranteed profit", "assured returns", "risk-free", "certain gains", or promising specific future percentage yields for investment, equity, mutual fund, or insurance products.
2. NO False Urgency or Pressure: Do not allow pressuring the customer with artificial deadlines (e.g., "within 5 minutes", "by end of day today") or threatening negative consequences like "account closure", "penalties", or "loss of benefits" unless contractually mandated.
3. NO Exaggerated or Misleading Promises: Do not allow claims of making the customer a "millionaire", doubling their wealth quickly, or factually unsupported performance statements.
4. Professional & Honest Tone: Ensure all claims are balanced, clear, and refer the customer to official terms.

Message to review: "{body}"

Respond ONLY with JSON, no markdown fences:
{{
  "compliant": true or false,
  "issues": ["list any compliance issues found, empty list if none"],
  "revised_body": "if not compliant, a corrected version. If compliant, repeat the original body unchanged."
}}"""


def run_validate(draft: dict, trace: list) -> dict:
    prompt = VALIDATE_PROMPT.format(body=draft.get("body", ""))
    for _attempt in range(len(_GROQ_KEYS) + 1):
        try:
            response = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            break  # success
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                if _attempt < len(_GROQ_KEYS):
                    rotate_key("429")
                    time.sleep(2)
                    continue
            raise  # non-quota error, re-raise immediately
    text = response.choices[0].message.content or ""
    text = clean_llm_json(text)
    try:
        validation = json.loads(text)
        if not isinstance(validation, dict):
            validation = {"compliant": True, "issues": [], "revised_body": draft.get("body", "")}
    except json.JSONDecodeError:
        validation = {"compliant": True, "issues": [], "revised_body": draft.get("body", "")}

    # Normalize validation keys
    if not validation.get("revised_body"):
        for key in ["revised_message", "revised_text", "body", "message", "text"]:
            if key in validation and validation[key]:
                validation["revised_body"] = validation[key]
                break

    trace.append({"step": "validate", "result": validation})

    if not validation.get("compliant", True) and validation.get("revised_body"):
        draft["body"] = validation["revised_body"]
        draft["was_revised"] = True
    else:
        draft["was_revised"] = False

    return draft


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def process_customer(event: dict, customer: dict) -> dict:
    trace = []
    trace.append({
        "step": "start",
        "result": {
            "customer_id": customer.get("customer_id"),
            "customer_name": customer.get("name"),
            "event_label": event.get("event_label"),
            "signal": event.get("signal"),
        },
    })

    decision = run_investigate_decide(event, customer, trace)
    draft = run_draft(event, customer, decision, trace)
    final_draft = run_validate(draft, trace)

    return {
        "type": "agent_decision",
        "customer_id": customer.get("customer_id"),
        "customer_name": customer.get("name"),
        "event_type": event.get("event_type"),
        "event_label": event.get("event_label"),
        "signal": event.get("signal"),
        "confidence": event.get("confidence"),
        "candidate_products": event.get("recommended_products", []),
        "final_products": decision.get("final_products") or [p["name"] for p in event.get("recommended_products", [])],
        "reasoning": decision.get("reasoning"),
        "priority": decision.get("priority"),
        "priority_reason": decision.get("priority_reason"),
        "outreach": final_draft,
        "trace": trace,
    }


def main():
    db.init_db() # Ensure tables exist
    events = load_json(EVENTS_FILE)
    customers = load_json(CUSTOMERS_FILE)
    customer_index = index_customers_by_id(customers)

    subset = pick_subset(events, MAX_CUSTOMERS)
    print(f"Processing {len(subset)} customers through the agentic pipeline (investigate -> decide -> draft -> validate)")
    print(f"Model: {MODEL_NAME} via Groq\n")

    DATA_DIR.mkdir(exist_ok=True)
    done_ids = set()
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM agent_runs")
        done_ids = {row["customer_id"] for row in cursor.fetchall()}
        if done_ids:
            print(f"Resuming — {len(done_ids)} already done.\n")
    except Exception as e:
        print(f"Error loading completed runs from database: {e}")
    finally:
        conn.close()

    inserted_count = 0
    for i, event in enumerate(subset, start=1):
        cust_id = event.get("customer_id")
        if cust_id in done_ids:
            print(f"[{i}/{len(subset)}] {cust_id}: already done, skipping")
            continue

        customer = customer_index.get(cust_id)
        if customer is None:
            print(f"[{i}/{len(subset)}] {cust_id}: SKIPPED (no customer profile)")
            continue

        print(f"[{i}/{len(subset)}] {cust_id} ({customer.get('name')}) — {event.get('event_label')}")
        try:
            result = process_customer(event, customer)
            tool_calls_made = [s["tool"] for s in result["trace"] if s["step"] == "tool_call"]
            print(f"  Tools called: {tool_calls_made or 'none'}")
            print(f"  Priority: {result['priority']} | Revised for compliance: {result['outreach'].get('was_revised')}")
            
            # Save directly to the database
            db.insert_agent_run(result)
            inserted_count += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        time.sleep(SECONDS_BETWEEN_CUSTOMERS)

    print(f"\nDone. {inserted_count} new agent traces saved to database.")


if __name__ == "__main__":
    main()
