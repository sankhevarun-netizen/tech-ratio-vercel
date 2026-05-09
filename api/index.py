"""
Tech Rationalization AI Agent — Vercel-ready, stateless FastAPI backend.
All state is managed by the client (browser). No server-side sessions.

Local dev:  uvicorn api.index:app --reload --port 8000
Vercel:     vercel deploy
"""

import json, uuid, io, os
from typing import Optional, List, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(title="Tech Rationalization AI Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve public/ folder as static (for local dev without Vercel CLI)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
if os.path.isdir(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

@app.get("/")
async def serve_ui():
    idx = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "API running. Open /static/index.html or use Vercel."}


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an Enterprise Technology Strategy AI Agent specializing in Platform, Application, and Tools Assessment & Rationalization, aligned with KPMG-style consulting frameworks and telecom/large enterprise transformation programs.

You combine the expertise of:
- Senior Enterprise Architect
- Technology Due-Diligence Consultant
- CIO/CTO Advisory Analyst
- Telecom Digital Transformation Expert (Airtel, Tanla-like environments)

## OBJECTIVE
Maximize business value, reduce cost, mitigate risk, and simplify the technology landscape by:
1. Assessing platforms, applications, and tools across 7 scoring dimensions
2. Applying the 6R Rationalization model
3. Detecting duplication and redundancy with consolidation paths
4. Generating executive-ready recommendations and roadmaps

## 7 SCORING DIMENSIONS (0–10 each)
1. Business Value — Strategic importance, revenue impact, criticality
2. Adoption Rate — User adoption %, utilization signals
3. Integration Depth — API dependencies, systemic coupling
4. Vendor Support — Roadmap clarity, vendor health, EOL status
5. Cost Efficiency — Cost-per-user vs market benchmarks
6. Technical Health — Modernity, tech debt, maintenance burden
7. Risk Score — Security, compliance, obsolescence, vendor lock-in (higher = more risky)

## 6R RATIONALIZATION MODEL
- Retain    → Strategic, healthy (Score ≥7.5, Risk ≤4)
- Rehost    → Lift-and-shift to cloud (Score ≥6, on-prem, cloud-ready)
- Replatform → Minor modernization (Score 5–7)
- Refactor  → Significant redesign (Score 3–5)
- Replace   → Better alternative exists (Score <5 or cost-inefficient)
- Retire    → Decommission — redundant, EOL, very low value (Score <3 or adoption <20%)

## TELECOM FILTERS (apply when context is telecom/TMT)
- Latency sensitivity, transaction volume at scale (millions/day), 24×7 SLA
- TRAI / GDPR / data sovereignty compliance
- Customer SLA impact, partner/vendor dependency risk

## DUPLICATION LOGIC
Flag tools in the same category with overlapping functionality. Quantify overlap %, identify retention candidate (higher score), estimate consolidation savings.
Example: "Tool A and Tool B show 78% functional overlap in APM. Tool B scores lower (5.2 vs 7.1) and costs more — Tool A is the strategic retention candidate."

## OUTPUT FORMAT
Always produce structured, executive-grade output with:
- Clear 6R decision + rationale
- Confidence level: High / Medium / Low
- Business impact, cost savings, risk reduction
- Phased roadmap: 0–3 months | 3–12 months | 12–24 months

Be data-driven. State assumptions explicitly. Provide trade-offs. Challenge poor decisions constructively. Never give generic recommendations."""


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE (inline — self-contained for Vercel)
# ═══════════════════════════════════════════════════════════════════════════
CAT_VALUE = {
    "BSS":9,"OSS":9,"CRM":8.5,"ERP":8.5,"Security":8.5,"Network":8,
    "Database":8,"Cloud":7.5,"Analytics":7.5,"ITSM":7,"Monitoring":6.5,
    "Logging":6.5,"APM":6.5,"DevOps":6,"Collaboration":5.5,"Storage":6,"Other":4.5,
}
CRIT_MOD = {"Critical":+2,"High":+1,"Medium":0,"Low":-1.5,None:0}
TIER1 = {"microsoft","aws","amazon","google","oracle","sap","salesforce","servicenow",
          "splunk","dynatrace","datadog","ibm","cisco","crowdstrike","elastic","palo alto"}

def score_tool(t: Dict) -> Dict[str, float]:
    # Business Value
    bv = min(10, max(0, CAT_VALUE.get(t.get("category","Other"),4.5) + CRIT_MOD.get(t.get("criticality"),0)))

    # Adoption Rate
    u = t.get("user_count") or 0
    ar = 5 if not u else (9.5 if u>=2000 else 9 if u>=1000 else 8 if u>=500 else
         7 if u>=200 else 6 if u>=100 else 5 if u>=50 else 4 if u>=20 else 3 if u>=10 else 2)

    # Integration Depth
    i = t.get("integrations") or 0
    id_ = 5 if not i else (9.5 if i>=25 else 8.5 if i>=15 else 7.5 if i>=10 else 6.5 if i>=5 else 5.5 if i>=3 else 4)

    # Vendor Support
    eol = bool(t.get("end_of_life"))
    vnd = (t.get("vendor") or "").lower()
    vs = 1 if eol else (8.5 if any(v in vnd for v in TIER1) else 6.5 if vnd else 5)

    # Cost Efficiency
    cost = t.get("annual_cost") or 0
    users = max(t.get("user_count") or 1, 1)
    if not cost: ce = 5
    else:
        cpu = cost / users
        ce = (9.5 if cpu<50 else 8.5 if cpu<200 else 7.5 if cpu<500 else 6 if cpu<1000
              else 4.5 if cpu<3000 else 3 if cpu<8000 else 1.5 if cpu<20000 else 0.5)

    # Technical Health
    age = t.get("age_years") or 0
    th = 1 if eol else (6 if not age else 9.5 if age<=1 else 8.5 if age<=3 else 7 if age<=5
         else 5.5 if age<=8 else 3.5 if age<=12 else 2.5 if age<=15 else 1.5)

    # Risk Score (higher = more risky)
    risk = 4.0
    if eol: risk += 3.5
    if t.get("compliance_required"): risk += 1
    if age > 12: risk += 2
    elif age > 8: risk += 1
    if t.get("criticality") == "Critical": risk += 0.5
    intp = t.get("integrations") or 0
    if intp >= 20: risk += 1
    elif intp >= 10: risk += 0.5
    if not vnd: risk += 0.5
    risk = min(10, max(0, risk))

    return {"business_value":round(bv,2),"adoption_rate":round(ar,2),
            "integration_depth":round(id_,2),"vendor_support":round(vs,2),
            "cost_efficiency":round(ce,2),"technical_health":round(th,2),
            "risk_score":round(risk,2)}

def composite_score(scores: Dict) -> float:
    w = {"business_value":0.25,"adoption_rate":0.15,"integration_depth":0.15,
         "vendor_support":0.15,"cost_efficiency":0.15,"technical_health":0.15}
    val = sum(scores.get(k,5)*wt for k,wt in w.items())
    penalty = max(0, (scores.get("risk_score",5)-5)*0.15)
    return round(min(10, max(0, val-penalty)), 2)

def action_6r(scores: Dict, t: Dict) -> str:
    c = composite_score(scores)
    risk = scores.get("risk_score",5)
    adopt = scores.get("adoption_rate",5)
    ce = scores.get("cost_efficiency",5)
    th = scores.get("technical_health",5)
    eol = bool(t.get("end_of_life"))
    dep = (t.get("deployment") or "").lower()
    if eol or c < 2.5: return "Retire"
    if c < 3.5 and adopt < 3: return "Retire"
    if c < 4.5 and ce < 3: return "Replace"
    if risk >= 8.5 and c < 6: return "Replace"
    if c >= 7.5 and risk <= 4: return "Retain"
    if c >= 6.5 and risk <= 5: return "Retain"
    if c >= 6 and "on-prem" in dep and th >= 5: return "Rehost"
    if c >= 5.5 and th < 5.5: return "Replatform"
    if c >= 4 and th < 4.5: return "Refactor"
    return "Retain" if c >= 6 else "Replatform"

def confidence_level(t: Dict) -> str:
    known = sum(1 for f in ["annual_cost","user_count","criticality","vendor","integrations","age_years","deployment"] if t.get(f))
    return "High" if known >= 5 else "Medium" if known >= 3 else "Low"


# ═══════════════════════════════════════════════════════════════════════════
#  DUPLICATION DETECTOR (inline)
# ═══════════════════════════════════════════════════════════════════════════
def detect_duplications(tools: List[Dict]) -> List[Dict]:
    by_cat: Dict[str, List] = defaultdict(list)
    for t in tools:
        by_cat[t.get("category","Other")].append(t)
    dups = []
    for cat, grp in by_cat.items():
        for i in range(len(grp)):
            for j in range(i+1, len(grp)):
                a, b = grp[i], grp[j]
                ov = _overlap(a, b)
                if ov >= 0.45:
                    dups.append(_dup_record(a, b, ov, cat))
    return sorted(dups, key=lambda x: x["overlap_percentage"], reverse=True)

def _overlap(a, b) -> float:
    s = 0.35
    if (a.get("subcategory") or "") == (b.get("subcategory") or "") and a.get("subcategory"): s += 0.25
    if (a.get("business_unit") or "").lower() == (b.get("business_unit") or "").lower() and a.get("business_unit"): s += 0.15
    va, vb = (a.get("vendor") or "").lower(), (b.get("vendor") or "").lower()
    if va and vb and va != vb: s += 0.10
    ua, ub = a.get("user_count") or 0, b.get("user_count") or 0
    if ua and ub and min(ua,ub)/max(ua,ub) > 0.5: s += 0.10
    return min(1.0, s)

def _dup_record(a, b, ov, cat) -> Dict:
    pct = round(ov * 100)
    sa, sb = a.get("composite_score",5), b.get("composite_score",5)
    retain, consol = (a,b) if sa >= sb else (b,a)
    rs, cs = max(sa,sb), min(sa,sb)
    cost_a, cost_b = a.get("annual_cost",0) or 0, b.get("annual_cost",0) or 0
    savings = round(min(cost_a, cost_b) * (ov/2))
    return {
        "id": f"dup-{a['id'][:8]}-{b['id'][:8]}",
        "category": cat,
        "tool_a": a.get("name","?"), "tool_b": b.get("name","?"),
        "overlap_percentage": pct,
        "retain_candidate": retain.get("name"),
        "consolidate_candidate": consol.get("name"),
        "potential_annual_savings": savings,
        "priority": "High" if pct>=70 else "Medium" if pct>=55 else "Low",
        "rationale": (f"{a.get('name')} and {b.get('name')} show {pct}% functional overlap in {cat}. "
                      f"{retain.get('name')} is the strategic retention candidate "
                      f"(score {rs:.1f} vs {cs:.1f}). "
                      f"Consolidation could yield ~${savings:,}/yr savings."),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DATA INGESTION (inline, in-memory — no disk writes for Vercel)
# ═══════════════════════════════════════════════════════════════════════════
CAT_KW = {
    "Monitoring":["monitor","prometheus","grafana","nagios","zabbix","pagerduty"],
    "Logging":["log","elk","elasticsearch","logstash","kibana","fluentd","splunk","sumo"],
    "APM":["apm","trace","dynatrace","newrelic","appdynamics","datadog","jaeger"],
    "Security":["security","siem","firewall","iam","pam","crowdstrike","qualys","veracode","snyk"],
    "ITSM":["itsm","incident","ticket","servicenow","jira","remedy","change"],
    "Collaboration":["collab","slack","teams","confluence","sharepoint","zoom","webex"],
    "CRM":["crm","salesforce","hubspot","siebel","dynamics"],
    "ERP":["erp","sap","workday","peoplesoft"],
    "BSS":["bss","billing","rating","charging"],
    "OSS":["oss","provisioning","inventory"],
    "Network":["network","sdwan","cisco","juniper","dns","routing","load balanc"],
    "Cloud":["cloud","aws","azure","gcp","openstack","terraform","kubernetes","k8s"],
    "Analytics":["analytic","bi","tableau","powerbi","qlik","looker","warehouse"],
    "DevOps":["devops","ci/cd","jenkins","gitlab","github","sonar","pipeline"],
    "Storage":["storage","backup","netapp","commvault"],
    "Database":["database","postgres","mysql","mongodb","redis","oracle db"],
}

COL_ALIASES = {
    "tool_name":"name","application":"name","app_name":"name","platform":"name","tool":"name",
    "vendor_name":"vendor","supplier":"vendor",
    "tool_category":"category","type":"category",
    "annual_license_cost":"annual_cost","cost_usd":"annual_cost","annual_spend":"annual_cost","cost":"annual_cost",
    "users":"user_count","num_users":"user_count","license_count":"user_count","active_users":"user_count",
    "team":"business_unit","department":"business_unit",
    "tool_owner":"owner","contact":"owner",
    "environment":"deployment","hosting":"deployment",
    "integration_count":"integrations","age":"age_years",
    "eol":"end_of_life","deprecated":"end_of_life",
}

def parse_csv_bytes(content: bytes) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(content))
    return _df_to_tools(df)

def parse_excel_bytes(content: bytes) -> List[Dict]:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(content))
    return _df_to_tools(df)

def parse_json_bytes(content: bytes) -> List[Dict]:
    data = json.loads(content)
    if isinstance(data, list): return [_norm(t) for t in data]
    for v in data.values():
        if isinstance(v, list): return [_norm(t) for t in v]
    return []

def parse_pdf_bytes(content: bytes) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages: text += page.extract_text() or ""
        return text[:8000]
    except Exception:
        return ""

def _df_to_tools(df) -> List[Dict]:
    import pandas as pd
    df.columns = [c.lower().strip().replace(" ","_").replace("-","_") for c in df.columns]
    df = df.rename(columns={k:v for k,v in COL_ALIASES.items() if k in df.columns})
    df = df.where(pd.notna(df), None)
    return [_norm(r.to_dict()) for _,r in df.iterrows()]

def _norm(d: Dict) -> Dict:
    def s(v): return "" if v is None else ("" if str(v).lower() in ("nan","none","null","n/a") else str(v).strip())
    def n(v):
        try: return float(str(v).replace(",","").replace("$","").replace("£","").replace("€","").strip())
        except: return None
    def ni(v): x=n(v); return int(x) if x is not None else None
    def b(v): return str(v).lower().strip() in ("true","yes","1","y","eol","deprecated")
    def cat(raw):
        low = raw.lower()
        for c,kws in CAT_KW.items():
            if any(k in low for k in kws): return c
        return raw.strip() or "Other"
    def dep(raw):
        low=raw.lower()
        if any(x in low for x in ["cloud","saas","paas","iaas"]): return "Cloud"
        if any(x in low for x in ["on-prem","onprem","premise"]): return "On-Prem"
        if "hybrid" in low: return "Hybrid"
        return raw or None
    def crit(v):
        if not v: return None
        x=str(v).lower()
        if "critical" in x: return "Critical"
        if "high" in x: return "High"
        if "med" in x: return "Medium"
        if "low" in x: return "Low"
        return str(v)
    t = {
        "id": str(uuid.uuid4()),
        "name": s(d.get("name") or d.get("tool_name") or "Unknown Tool"),
        "vendor": s(d.get("vendor")) or None,
        "category": cat(s(d.get("category",""))),
        "description": s(d.get("description")) or None,
        "owner": s(d.get("owner")) or None,
        "business_unit": s(d.get("business_unit")) or None,
        "annual_cost": n(d.get("annual_cost")),
        "user_count": ni(d.get("user_count")),
        "license_type": s(d.get("license_type")) or None,
        "deployment": dep(s(d.get("deployment",""))),
        "criticality": crit(d.get("criticality")),
        "integrations": ni(d.get("integrations")),
        "age_years": n(d.get("age_years")),
        "end_of_life": b(d.get("end_of_life",False)),
        "compliance_required": b(d.get("compliance_required",False)),
    }
    return {k:v for k,v in t.items() if v not in (None,"") or k in ("id","name","category","end_of_life")}

def apply_scores(tools: List[Dict]) -> List[Dict]:
    for t in tools:
        sc = score_tool(t)
        t["scores"] = sc
        t["composite_score"] = composite_score(sc)
        t["rationalization_action"] = action_6r(sc, t)
        t["confidence_level"] = confidence_level(t)
    return tools


# ═══════════════════════════════════════════════════════════════════════════
#  AI HELPERS
# ═══════════════════════════════════════════════════════════════════════════
async def ai_parse_text(text: str) -> List[Dict]:
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":
            f"""Extract all tools/applications/platforms from the text below.
Return a JSON object with a "tools" array. Each item must have:
  name, vendor (or null), category (use: Monitoring/Logging/APM/Security/ITSM/Collaboration/
  CRM/ERP/BSS/OSS/Cloud/Analytics/DevOps/Network/Storage/Database/Other),
  description (brief, or null), annual_cost (number or null), user_count (number or null),
  criticality ("Critical"|"High"|"Medium"|"Low"|null),
  deployment ("Cloud"|"On-Prem"|"Hybrid"|null), integrations (number or null),
  age_years (number or null), end_of_life (bool), compliance_required (bool),
  business_unit (string or null).
TEXT:\n{text}"""}],
        max_tokens=3000, temperature=0.1,
        response_format={"type":"json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        if isinstance(data, list): return data
        for v in data.values():
            if isinstance(v, list): return v
        return []
    except: return []

async def ai_assess(tools: List[Dict], dups: List[Dict], industry: str, focus: str) -> Dict:
    prompt = f"""Perform a comprehensive rationalization assessment.
Industry: {industry.upper()}{(' | Focus: ' + focus) if focus else ''}

TOOL INVENTORY (pre-scored):
{json.dumps(tools, indent=2)[:8000]}

DUPLICATION ANALYSIS:
{json.dumps(dups, indent=2)[:2000]}

Return ONLY valid JSON with these exact keys:
{{
  "executive_summary": "<3–5 paragraph CIO-ready narrative>",
  "portfolio_overview": {{
    "total_tools": <int>, "total_annual_cost": <float>,
    "portfolio_health": "Healthy|At Risk|Critical",
    "health_rationale": "<brief>"
  }},
  "rationalization_summary": {{"Retain":<int>,"Rehost":<int>,"Replatform":<int>,"Refactor":<int>,"Replace":<int>,"Retire":<int>}},
  "top_recommendations": [
    {{"rank":1,"title":"<title>","description":"<detail>","impact":"<impact>",
      "effort":"Low|Medium|High","priority":"Critical|High|Medium",
      "confidence":"High|Medium|Low","timeline":"0-3 months|3-12 months|12-24 months"}}
  ],
  "consolidation_opportunities": [
    {{"tools":["A","B"],"category":"<cat>","overlap_pct":<int>,
      "recommended_action":"<action>","estimated_savings":<float>,"rationale":"<why>"}}
  ],
  "risk_highlights": [
    {{"risk_type":"Security|Compliance|Vendor|Obsolescence|Operational",
      "severity":"Critical|High|Medium","affected_tools":["<names>"],
      "description":"<desc>","mitigation":"<action>"}}
  ],
  "roadmap": {{
    "short_term": ["<0-3 month action>"],
    "medium_term": ["<3-12 month action>"],
    "long_term": ["<12-24 month action>"]
  }},
  "expected_outcomes": {{
    "cost_savings_annual": <float>,
    "risk_reduction": "<desc>",
    "tool_reduction": "<from X to Y>",
    "strategic_gains": "<value>"
  }}
}}"""
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        max_tokens=6000, temperature=0.2,
        response_format={"type":"json_object"},
    )
    try: return json.loads(resp.choices[0].message.content)
    except: return {"executive_summary": resp.choices[0].message.content}

def build_report_html(tools: List[Dict], dups: List[Dict], assessment: Dict) -> str:
    from datetime import datetime
    total_cost = sum(t.get("annual_cost",0) or 0 for t in tools)
    pot_savings = sum(d.get("potential_annual_savings",0) or 0 for d in dups)
    action_counts: Dict[str,int] = {}
    for t in tools:
        a = t.get("rationalization_action","TBD")
        action_counts[a] = action_counts.get(a,0)+1

    ACTION_DESC = {
        "Retain":"Strategic & healthy — no immediate change","Rehost":"Lift-and-shift to cloud",
        "Replatform":"Minor modernization with managed services","Refactor":"Significant redesign required",
        "Replace":"Better alternative exists — migrate","Retire":"Decommission — redundant or low-value",
    }
    def badge(action):
        cls = action.lower()
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;{badge_style(action)}">{action}</span>'
    def badge_style(a):
        styles = {
            "Retain":"color:#155724;background:#d4edda","Rehost":"color:#004085;background:#cce5ff",
            "Replatform":"color:#856404;background:#fff3cd","Refactor":"color:#7d3c00;background:#fde8d8",
            "Replace":"color:#721c24;background:#f8d7da","Retire":"color:#383d41;background:#e2e3e5",
            "High":"color:#721c24;background:#f8d7da","Medium":"color:#856404;background:#fff3cd",
            "Low":"color:#155724;background:#d4edda","Critical":"color:#fff;background:#c0392b",
        }
        return styles.get(a,"color:#333;background:#eee")

    tool_rows = "".join(f"""<tr>
      <td><strong>{t.get('name','')}</strong></td><td>{t.get('vendor') or '—'}</td>
      <td>{t.get('category','')}</td><td>{'$'+f"{t.get('annual_cost',0):,.0f}" if t.get('annual_cost') else '—'}</td>
      <td>{t.get('user_count') or '—'}</td>
      <td><strong>{t.get('composite_score','—')}</strong>/10</td>
      <td>{t.get('scores',{}).get('risk_score','—')}/10</td>
      <td>{badge(t.get('rationalization_action','TBD'))}</td></tr>""" for t in tools)

    dup_rows = "".join(f"""<tr>
      <td>{d.get('category')}</td><td><b>{d.get('tool_a')}</b></td><td><b>{d.get('tool_b')}</b></td>
      <td><b>{d.get('overlap_percentage')}%</b></td><td>{d.get('retain_candidate')}</td>
      <td>${d.get('potential_annual_savings',0):,.0f}</td>
      <td>{badge(d.get('priority','Low'))}</td></tr>""" for d in dups[:15])

    action_rows = "".join(f"""<tr>
      <td>{badge(a)}</td><td><b>{c}</b></td>
      <td>{round(c/max(sum(action_counts.values()),1)*100)}%</td>
      <td>{ACTION_DESC.get(a,'')}</td></tr>""" for a,c in action_counts.items())

    exec_sum = assessment.get("executive_summary","Assessment not yet run.")
    recs = assessment.get("top_recommendations",[])
    roadmap = assessment.get("roadmap",{})
    outcomes = assessment.get("expected_outcomes",{})
    risks = assessment.get("risk_highlights",[])

    rec_html = "".join(f"""<div style="background:#f8fbff;border:1px solid #e0e8f8;border-left:4px solid #0063DC;
        border-radius:8px;padding:18px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:26px;height:26px;background:#0063DC;color:#fff;border-radius:50%;
          display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0">{r.get('rank','')}</div>
        <strong style="color:#003366">{r.get('title','')}</strong>
        <span style="margin-left:auto;display:flex;gap:6px">
          {badge(r.get('priority','Medium'))} {badge(r.get('confidence','Medium'))}
        </span>
      </div>
      <p style="font-size:13px;color:#444;line-height:1.6;margin-bottom:6px">{r.get('description','')}</p>
      <p style="font-size:12px;color:#0063DC;font-style:italic">Impact: {r.get('impact','')} &bull; {r.get('timeline','')}</p>
    </div>""" for r in recs[:5])

    def phase_items(key):
        items = roadmap.get(key,[])
        if isinstance(items,str): items=[items]
        return "".join(f'<li style="padding:6px 0;border-bottom:1px solid rgba(0,0,0,.06)">{x}</li>' for x in items)

    outcome_rows = "".join(f"<tr><td><strong>{k.replace('_',' ').title()}</strong></td><td>{v}</td></tr>" for k,v in outcomes.items())

    risk_html = "".join(f"""<div style="display:flex;gap:14px;background:#fff5f5;border:1px solid #fad7d7;
        border-radius:10px;padding:16px;margin-bottom:10px">
      <div style="font-size:20px">{'🔴' if r.get('severity')=='Critical' else '🟠' if r.get('severity')=='High' else '🟡'}</div>
      <div>
        <div style="font-weight:700;color:#1a2340">{r.get('risk_type')} Risk — {r.get('severity')}</div>
        <div style="font-size:13px;color:#666;margin:4px 0">{r.get('description','')}</div>
        <div style="font-size:12px;color:#00A651">Mitigation: {r.get('mitigation','')}</div>
      </div></div>""" for r in risks)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Tech Rationalization Report — {datetime.now().strftime("%B %Y")}</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4fa;color:#1a2340;font-size:14px}}
.report{{max-width:1200px;margin:0 auto;background:#fff;box-shadow:0 0 40px rgba(0,0,0,.1)}}
.header{{background:linear-gradient(135deg,#003366,#0063DC);color:#fff;padding:40px}}
.header h1{{font-size:24px;font-weight:700}}
.header p{{opacity:.75;font-size:13px;margin-top:4px}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#e0e7f0}}
.kpi-card{{background:#fff;padding:24px;text-align:center}}
.kpi-val{{font-size:30px;font-weight:800;color:#0063DC}}
.kpi-lbl{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.8px;margin-top:4px}}
.section{{padding:32px 40px;border-bottom:1px solid #eef1f8}}
h2{{font-size:17px;font-weight:700;color:#003366;margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #0063DC}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{background:#003366;color:#fff;padding:10px 14px;text-align:left;font-size:12px;font-weight:600}}
tbody td{{padding:9px 14px;border-bottom:1px solid #f0f3fa}}
tbody tr:hover{{background:#f8fbff}}
.exec{{background:#f8fbff;border-left:4px solid #0063DC;padding:20px;border-radius:0 8px 8px 0;
  font-size:13px;line-height:1.8;white-space:pre-wrap}}
.roadmap{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.phase{{background:#f8fbff;border-radius:8px;padding:18px}}
.phase ul{{list-style:none}}
.footer{{background:#1a2340;color:#8899bb;padding:18px 40px;display:flex;justify-content:space-between;font-size:12px}}
@media print{{body{{background:#fff}}.report{{box-shadow:none}}.roadmap{{grid-template-columns:repeat(3,1fr)}}}}
</style></head><body>
<div class="report">
<div class="header">
  <h1>Technology Rationalization Assessment Report</h1>
  <p>Enterprise Platform, Application &amp; Tools Assessment &bull; AI-Powered Advisory &bull; KPMG-Style 6R Framework</p>
  <p style="margin-top:10px;opacity:.6;font-size:12px">Generated: {datetime.now().strftime("%d %B %Y %H:%M")} &bull; CONFIDENTIAL</p>
</div>
<div class="kpi">
  <div class="kpi-card"><div class="kpi-val">{len(tools)}</div><div class="kpi-lbl">Tools Assessed</div></div>
  <div class="kpi-card"><div class="kpi-val">${total_cost:,.0f}</div><div class="kpi-lbl">Annual Spend</div></div>
  <div class="kpi-card"><div class="kpi-val">{len(dups)}</div><div class="kpi-lbl">Overlap Pairs</div></div>
  <div class="kpi-card"><div class="kpi-val">${pot_savings:,.0f}</div><div class="kpi-lbl">Est. Savings</div></div>
</div>
<div class="section"><h2>Executive Summary</h2><div class="exec">{exec_sum}</div></div>
<div class="section"><h2>Rationalization Action Summary</h2>
<table><thead><tr><th>6R Action</th><th>Count</th><th>% of Portfolio</th><th>Description</th></tr></thead>
<tbody>{action_rows}</tbody></table></div>
{"<div class='section'><h2>Top Priority Recommendations</h2>" + rec_html + "</div>" if rec_html else ""}
{f"""<div class="section"><h2>Rationalization Roadmap</h2>
<div class="roadmap">
  <div class="phase" style="border-top:3px solid #00A651"><h3 style="color:#00A651;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:12px">Phase 1 — Quick Wins (0–3 Months)</h3><ul>{phase_items('short_term')}</ul></div>
  <div class="phase" style="border-top:3px solid #0063DC"><h3 style="color:#0063DC;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:12px">Phase 2 — Strategic (3–12 Months)</h3><ul>{phase_items('medium_term')}</ul></div>
  <div class="phase" style="border-top:3px solid #7B2FBE"><h3 style="color:#7B2FBE;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:12px">Phase 3 — Transformation (12–24 Months)</h3><ul>{phase_items('long_term')}</ul></div>
</div></div>""" if roadmap else ""}
{f"<div class='section'><h2>Risk Highlights</h2>{risk_html}</div>" if risk_html else ""}
<div class="section"><h2>Full Tool Portfolio Assessment</h2>
<table><thead><tr><th>Tool</th><th>Vendor</th><th>Category</th><th>Annual Cost</th><th>Users</th><th>Score</th><th>Risk</th><th>Action</th></tr></thead>
<tbody>{tool_rows}</tbody></table></div>
{f"""<div class="section"><h2>Duplication &amp; Consolidation Opportunities</h2>
<table><thead><tr><th>Category</th><th>Tool A</th><th>Tool B</th><th>Overlap</th><th>Retain</th><th>Est. Savings</th><th>Priority</th></tr></thead>
<tbody>{dup_rows}</tbody></table></div>""" if dups else ""}
{f"""<div class="section"><h2>Expected Business Outcomes</h2>
<table><tbody>{outcome_rows}</tbody></table></div>""" if outcomes else ""}
<div class="footer">
  <span>Tech Rationalization AI Agent &bull; Enterprise Technology Strategy Advisory</span>
  <span>CONFIDENTIAL &bull; {datetime.now().strftime('%Y')}</span>
</div></div></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════
class ChatReq(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    tools: List[Dict] = []

class AssessReq(BaseModel):
    tools: List[Dict]
    duplications: List[Dict] = []
    industry: str = "telecom"
    focus: Optional[str] = None

class ReportReq(BaseModel):
    tools: List[Dict]
    duplications: List[Dict] = []
    assessment: Dict = {}


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/chat")
async def chat(req: ChatReq):
    msgs = [{"role":"system","content":SYSTEM_PROMPT}]
    if req.tools:
        summary = "\n".join(
            f"- {t.get('name')} | {t.get('category')} | Action:{t.get('rationalization_action','TBD')} | Score:{t.get('composite_score','?')}"
            for t in req.tools[:30]
        )
        msgs.append({"role":"system","content":f"Current tool inventory ({len(req.tools)} tools):\n{summary}"})
    for m in req.history[-20:]:
        msgs.append({"role": m.get("role","user"), "content": m.get("content","")})
    msgs.append({"role":"user","content":req.message})
    resp = await client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, max_tokens=2000, temperature=0.2)
    return {"reply": resp.choices[0].message.content}


@app.post("/api/ingest")
async def ingest(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    tools: List[Dict] = []
    if file and file.filename:
        content = await file.read()
        ext = file.filename.lower().rsplit(".",1)[-1]
        if ext == "csv":       tools = parse_csv_bytes(content)
        elif ext in ("xlsx","xls"): tools = parse_excel_bytes(content)
        elif ext == "json":    tools = parse_json_bytes(content)
        elif ext == "pdf":
            raw = parse_pdf_bytes(content)
            if raw: tools = [_norm(t) for t in await ai_parse_text(raw)]
    elif text:
        tools = [_norm(t) for t in await ai_parse_text(text)]

    tools = apply_scores(tools)
    dups  = detect_duplications(tools)
    return {
        "tools": tools,
        "duplications": dups,
        "summary": {
            "total_tools": len(tools),
            "total_annual_cost": sum(t.get("annual_cost",0) or 0 for t in tools),
            "duplications_found": len(dups),
            "potential_savings": sum(d.get("potential_annual_savings",0) for d in dups),
        }
    }


@app.post("/api/assess")
async def assess(req: AssessReq):
    if not req.tools:
        raise HTTPException(400, "No tools provided.")
    result = await ai_assess(req.tools, req.duplications, req.industry, req.focus or "")
    return {"assessment": result}


@app.post("/api/report")
async def report(req: ReportReq):
    html = build_report_html(req.tools, req.duplications, req.assessment)
    return {"html": html}


@app.get("/api/health")
async def health():
    return {"status":"healthy","model":OPENAI_MODEL}
