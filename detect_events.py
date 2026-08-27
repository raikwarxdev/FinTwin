"""
FinTwin — Life Event Detection Engine
Dev Kumar Raikwar

Reads transaction data and detects meaningful life events using
rule-based logic + statistical thresholds. Outputs a JSON file
of detected events with confidence scores and recommended SBI products.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

INPUT_TRANSACTIONS = "data/transactions.csv"
INPUT_CUSTOMERS    = "data/customers.json"
OUTPUT_EVENTS      = "data/detected_events.json"

# SBI product catalogue (simplified for prototype)
SBI_PRODUCTS = {
    "salary_jump": [
        {"name": "SBI Salary Account Upgrade", "type": "account"},
        {"name": "SBI Personal Loan (Pre-approved)", "type": "loan"},
        {"name": "SBI SIP — Magnum Equity Fund", "type": "investment"},
    ],
    "new_emi": [
        {"name": "SBI Home Loan Balance Transfer", "type": "loan"},
        {"name": "SBI Loan Protection Insurance", "type": "insurance"},
        {"name": "SBI Overdraft Facility", "type": "credit"},
    ],
    "savings_milestone": [
        {"name": "SBI Fixed Deposit (High Yield)", "type": "investment"},
        {"name": "SBI Mutual Fund — Bluechip", "type": "investment"},
        {"name": "SBI Life Smart Wealth Plan", "type": "insurance"},
    ],
    "large_withdrawal": [
        {"name": "SBI Overdraft Facility", "type": "credit"},
        {"name": "SBI Personal Loan (Pre-approved)", "type": "loan"},
        {"name": "SBI Emergency Credit Line", "type": "credit"},
    ],
    "retirement_approaching": [
        {"name": "SBI Life — Smart Pension Plan", "type": "insurance"},
        {"name": "SBI Annuity Deposit Scheme", "type": "investment"},
        {"name": "SBI Senior Citizens Savings Scheme", "type": "investment"},
    ],
}


def load_data():
    df = pd.read_csv(INPUT_TRANSACTIONS)
    with open(INPUT_CUSTOMERS) as f:
        customers = {c["customer_id"]: c for c in json.load(f)}
    return df, customers


def get_monthly_salary(df, cid):
    """Returns monthly salary credits for a customer."""
    sal = df[(df["customer_id"] == cid) & (df["category"] == "salary")]
    return sal.groupby("month")["amount"].sum()


def get_monthly_emi(df, cid):
    """Returns monthly EMI debits for a customer."""
    emi = df[(df["customer_id"] == cid) & (df["category"] == "emi")]
    return emi.groupby("month")["amount"].sum()


def get_monthly_savings(df, cid):
    """Returns monthly savings transfers for a customer."""
    sav = df[(df["customer_id"] == cid) & (df["category"] == "savings_transfer")]
    return sav.groupby("month")["amount"].sum()


# ---------- Detectors ----------

def detect_salary_jump(salary_series, threshold=0.35):
    """
    Detects a salary jump of >35% between consecutive months.
    Returns (detected, month_detected, confidence, old_salary, new_salary)
    """
    if len(salary_series) < 2:
        return False, None, 0, None, None

    months = sorted(salary_series.index)
    for i in range(1, len(months)):
        prev = salary_series[months[i - 1]]
        curr = salary_series[months[i]]
        if prev > 0 and (curr - prev) / prev >= threshold:
            jump_pct = (curr - prev) / prev
            # Confidence: higher jump = higher confidence, capped at 0.98
            confidence = min(0.98, 0.60 + jump_pct * 0.5)
            return True, months[i], round(confidence, 2), int(prev), int(curr)

    return False, None, 0, None, None


def detect_new_emi(emi_series, threshold=5000):
    """
    Detects a new recurring EMI appearing mid-simulation.
    Looks for a new debit category that wasn't present in month 1
    and is above threshold.
    """
    if len(emi_series) < 2:
        return False, None, 0, None

    months = sorted(emi_series.index)
    baseline = emi_series[months[0]] if months[0] == 1 else 0

    for m in months[1:]:
        curr = emi_series[m]
        jump = curr - baseline
        if jump >= threshold:
            confidence = min(0.97, 0.55 + (jump / 50000) * 0.4)
            return True, m, round(confidence, 2), int(jump)

    return False, None, 0, None


def detect_savings_milestone(savings_series, milestones=[100000, 200000, 500000]):
    """
    Detects when cumulative savings cross a meaningful milestone.
    """
    cumulative = 0
    months = sorted(savings_series.index)

    for m in months:
        cumulative += savings_series[m]
        for milestone in milestones:
            if cumulative >= milestone * 0.92:  # within 8% of milestone
                # Confidence based on how close to exact milestone
                ratio = cumulative / milestone
                confidence = min(0.96, 0.70 + (1 - abs(1 - ratio)) * 0.26)
                return True, m, round(confidence, 2), milestone

    return False, None, 0, None



def detect_large_withdrawal(df_customer, c):
    """Detects an injected large withdrawal transaction marked by description."""
    if "description" not in df_customer.columns:
        return False, 0, 0
    flagged = df_customer[df_customer["description"] == "LARGE WITHDRAWAL"]
    if len(flagged) > 0:
        row = flagged.iloc[-1]
        salary = df_customer[df_customer["category"] == "salary"]["amount"].mean() or 30000
        confidence = min(0.92, 0.65 + (row["amount"] / salary) * 0.1)
        return True, round(confidence, 2), int(row["amount"])
    return False, 0, 0


def detect_retirement_approaching(c, df_customer=None):
    """Detects an injected retirement-savings transaction marked by description."""
    if df_customer is None or "description" not in df_customer.columns:
        return False, 0
    flagged = df_customer[df_customer["description"] == "RETIREMENT SAVINGS"]
    if len(flagged) > 0:
        age = c.get("age", 55)
        if age < 50:
            return False, 0
        confidence = round(min(0.95, 0.70 + max(0, age - 52) * 0.03), 2)
        return True, confidence
    return False, 0


# ---------- Main detector ----------

def detect_customer_events(df_customer, c):
    """
    Runs event detection rules for a single customer's transactions.
    df_customer: DataFrame containing transactions ONLY for this customer.
    c: customer profile dictionary.
    """
    detected = []
    cid = c.get("customer_id")
    if not cid:
        return detected

    # --- Large withdrawal ---
    found, conf, amt = detect_large_withdrawal(df_customer, c)
    if found and conf >= 0.60:
        detected.append({
            "customer_id": cid,
            "customer_name": c.get("name", "Unknown"),
            "age": c.get("age"),
            "city": c.get("city"),
            "event_type": "large_withdrawal",
            "event_label": "Large Withdrawal Detected",
            "detected_month": 1,
            "confidence": conf,
            "signal": f"Large withdrawal of ₹{amt:,} detected — may indicate financial stress or major purchase",
            "old_value": 0,
            "new_value": amt,
            "recommended_products": SBI_PRODUCTS["large_withdrawal"],
            "detected_at": datetime.now().isoformat(),
        })

    # --- Retirement approaching ---
    found, conf = detect_retirement_approaching(c, df_customer)
    if found:
        detected.append({
            "customer_id": cid,
            "customer_name": c.get("name", "Unknown"),
            "age": c.get("age"),
            "city": c.get("city"),
            "event_type": "retirement_approaching",
            "event_label": "Retirement Planning Opportunity",
            "detected_month": 1,
            "confidence": conf,
            "signal": f"Customer aged {c.get('age')} — ideal time to review retirement and pension plans",
            "old_value": 0,
            "new_value": c.get("age"),
            "recommended_products": SBI_PRODUCTS["retirement_approaching"],
            "detected_at": datetime.now().isoformat(),
        })

    # Helper to group by month
    salary_series = df_customer[df_customer["category"] == "salary"].groupby("month")["amount"].sum()
    emi_series    = df_customer[df_customer["category"] == "emi"].groupby("month")["amount"].sum()
    sav_series    = df_customer[df_customer["category"] == "savings_transfer"].groupby("month")["amount"].sum()

    # --- Salary jump ---
    found, month, conf, old_sal, new_sal = detect_salary_jump(salary_series)
    if found and conf >= 0.65:
        detected.append({
            "customer_id": cid,
            "customer_name": c.get("name", "Unknown"),
            "age": c.get("age"),
            "city": c.get("city"),
            "event_type": "salary_jump",
            "event_label": "New Job Detected",
            "detected_month": int(month),
            "confidence": conf,
            "signal": f"Salary increased from ₹{old_sal:,} to ₹{new_sal:,} ({round((new_sal-old_sal)/old_sal*100)}% jump)",
            "old_value": old_sal,
            "new_value": new_sal,
            "recommended_products": SBI_PRODUCTS["salary_jump"],
            "detected_at": datetime.now().isoformat(),
        })

    # --- New EMI ---
    found, month, conf, emi_amt = detect_new_emi(emi_series)
    if found and conf >= 0.60:
        detected.append({
            "customer_id": cid,
            "customer_name": c.get("name", "Unknown"),
            "age": c.get("age"),
            "city": c.get("city"),
            "event_type": "new_emi",
            "event_label": "New Financial Commitment",
            "detected_month": int(month),
            "confidence": conf,
            "signal": f"New recurring EMI of ₹{emi_amt:,}/month detected",
            "old_value": 0,
            "new_value": emi_amt,
            "recommended_products": SBI_PRODUCTS["new_emi"],
            "detected_at": datetime.now().isoformat(),
        })

    # --- Savings milestone ---
    found, month, conf, milestone = detect_savings_milestone(sav_series)
    if found and conf >= 0.70:
        detected.append({
            "customer_id": cid,
            "customer_name": c.get("name", "Unknown"),
            "age": c.get("age"),
            "city": c.get("city"),
            "event_type": "savings_milestone",
            "event_label": "Savings Milestone Reached",
            "detected_month": int(month),
            "confidence": conf,
            "signal": f"Cumulative savings approaching ₹{milestone:,} milestone",
            "old_value": 0,
            "new_value": milestone,
            "recommended_products": SBI_PRODUCTS["savings_milestone"],
            "detected_at": datetime.now().isoformat(),
        })

    return detected


def detect_all_events(df, customers):
    detected = []
    customer_ids = df["customer_id"].unique()

    print(f"Running detection on {len(customer_ids)} customers...")

    for cid in customer_ids:
        c = customers.get(cid, {})
        df_cust = df[df["customer_id"] == cid]
        detected.extend(detect_customer_events(df_cust, c))

    # Sort by confidence descending
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    return detected


def print_summary(detected):
    print(f"\n{'='*55}")
    print(f"  FINTWIN — LIFE EVENT DETECTION RESULTS")
    print(f"{'='*55}")
    print(f"  Total events detected: {len(detected)}")

    by_type = {}
    for d in detected:
        t = d["event_type"]
        by_type[t] = by_type.get(t, 0) + 1

    for t, count in by_type.items():
        label = {
            "salary_jump": "New Job / Salary Jump",
            "new_emi":     "New EMI / Loan",
            "savings_milestone": "Savings Milestone",
        }.get(t, t)
        print(f"  — {label:<28} {count}")

    print(f"\n  Top 5 highest-confidence detections:")
    print(f"  {'Customer':<20} {'Event':<25} {'Conf':>6}  Signal")
    print(f"  {'-'*80}")
    for d in detected[:5]:
        name = d["customer_name"][:18]
        event = d["event_label"][:23]
        conf = f"{d['confidence']*100:.0f}%"
        signal = d["signal"][:45]
        print(f"  {name:<20} {event:<25} {conf:>6}  {signal}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    df, customers = load_data()
    detected = detect_all_events(df, customers)

    with open(OUTPUT_EVENTS, "w") as f:
        json.dump(detected, f, indent=2)

    print_summary(detected)
    print(f"✓ Detected events saved to {OUTPUT_EVENTS}")
