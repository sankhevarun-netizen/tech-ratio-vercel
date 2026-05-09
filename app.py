"""
Tech Rationalization AI Agent — Single File, VS Code Ready
=========================================================
SETUP:
  1. pip install fastapi uvicorn openai python-dotenv pandas openpyxl pdfplumber python-multipart
  2. Create .env file with:  OPENAI_API_KEY=sk-...
  3. python app.py
  4. Browser opens at http://localhost:8000
"""

import os, json, uuid, io, webbrowser, threading
from typing import Optional, List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import uvicorn

load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an Enterprise Technology Strategy AI Agent specializing in Platform, Application, and Tools Assessment & Rationalization, aligned with KPMG-style consulting frameworks and telecom/large enterprise transformation programs (e.g., Airtel, Tanla-like environments).

You combine the expertise of:
- Senior Enterprise Architect
- Technology Due-Diligence Consultant
- CIO/CTO Advisory Analyst
- Telecom Digital Transformation Expert

PRIMARY OBJECTIVE: Maximize business value, reduce cost, mitigate risk, and simplify the technology landscape by assessing tools across 7 dimensions and applying the 6R Rationalization model.

6R MODEL:
- Retain    → Strategic, healthy, high-value (Score ≥7.5, Risk ≤4)
- Rehost    → Lift-and-shift to cloud (Score ≥6, on-prem, cloud-ready)
- Replatform → Minor modernization (Score 5–7)
- Refactor  → Significant redesign needed (Score 3–5)
- Replace   → Better alternative exists (Score <5 or cost-inefficient)
- Retire    → Decommission — redundant, EOL, or very low value

SCORING DIMENSIONS (0–10 each):
1. Business Value — Strategic importance, revenue impact, criticality
2. Adoption Rate — User adoption %, utilization signals
3. Integration Depth — API dependencies, systemic coupling
4. Vendor Support — Roadmap clarity, vendor health, EOL status
5. Cost Efficiency — Cost-per-user vs market benchmarks
6. Technical Health — Modernity, tech debt, maintenance burden
7. Risk Score — Security, compliance, obsolescence, vendor lock-in (higher = more risky)

TELECOM FILTERS (apply when telecom/TMT context): Latency sensitivity, 24x7 SLA, TRAI/GDPR compliance, transaction volume at scale, customer SLA impact.

BEHAVIOR: Be data-driven. State assumptions explicitly. Tag recommendations with confidence: High/Medium/Low. Provide trade-offs, not just conclusions. Use executive-grade language. Never give generic recommendations.

OUTPUT: Always produce structured recommendations with rationale, impact analysis, roadmap (0-3 months | 3-12 months | 12-24 months), and expected outcomes."""

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
CAT_VALUE = {
    "BSS":9,"OSS":9,"CRM":8.5,"ERP":8.5,"Security":8.5,"Network":8,"Database":8,
    "Cloud":7.5,"Analytics":7.5,"ITSM":7,"Monitoring":6.5,"Logging":6.5,"APM":6.5,
    "DevOps":6,"Collaboration":5.5,"Storage":6,"Other":4.5,
}
CRIT_MOD  = {"Critical":2,"High":1,"Medium":0,"Low":-1.5,None:0}
TIER1_VND = {"microsoft","aws","amazon","google","oracle","sap","salesforce","servicenow",
             "splunk","dynatrace","datadog","ibm","cisco","crowdstrike","elastic","palo alto"}

def score_tool(t: Dict) -> Dict[str, float]:
    eol  = bool(t.get("end_of_life"))
    vnd  = (t.get("vendor") or "").lower()
    age  = t.get("age_years") or 0
    u    = t.get("user_count") or 0
    ints = t.get("integrations") or 0
    cost = t.get("annual_cost") or 0

    bv = min(10, max(0, CAT_VALUE.get(t.get("category","Other"),4.5) + CRIT_MOD.get(t.get("criticality"),0)))
    ar = 5 if not u else (9.5 if u>=2000 else 9 if u>=1000 else 8 if u>=500 else 7 if u>=200 else
         6 if u>=100 else 5 if u>=50 else 4 if u>=20 else 3 if u>=10 else 2)
    id_= 5 if not ints else (9.5 if ints>=25 else 8.5 if ints>=15 else 7.5 if ints>=10 else
         6.5 if ints>=5 else 5.5 if ints>=3 else 4)
    vs = 1 if eol else (8.5 if any(v in vnd for v in TIER1_VND) else 6.5 if vnd else 5)
    cpu = cost / max(u,1) if cost else 0
    ce  = 5 if not cost else (9.5 if cpu<50 else 8.5 if cpu<200 else 7.5 if cpu<500 else
          6 if cpu<1000 else 4.5 if cpu<3000 else 3 if cpu<8000 else 1.5 if cpu<20000 else 0.5)
    th  = 1 if eol else (6 if not age else 9.5 if age<=1 else 8.5 if age<=3 else 7 if age<=5 else
          5.5 if age<=8 else 3.5 if age<=12 else 2.5 if age<=15 else 1.5)
    risk = 4.0
    if eol: risk += 3.5
    if t.get("compliance_required"): risk += 1
    if age > 12: risk += 2
    elif age > 8: risk += 1
    if t.get("criticality") == "Critical": risk += 0.5
    if ints >= 20: risk += 1
    elif ints >= 10: risk += 0.5
    if not vnd: risk += 0.5
    return {"business_value":round(bv,2),"adoption_rate":round(ar,2),"integration_depth":round(id_,2),
            "vendor_support":round(vs,2),"cost_efficiency":round(ce,2),"technical_health":round(th,2),
            "risk_score":round(min(10,max(0,risk)),2)}

def composite(scores: Dict) -> float:
    w = {"business_value":0.25,"adoption_rate":0.15,"integration_depth":0.15,
         "vendor_support":0.15,"cost_efficiency":0.15,"technical_health":0.15}
    val = sum(scores.get(k,5)*wt for k,wt in w.items())
    return round(min(10, max(0, val - max(0,(scores.get("risk_score",5)-5)*0.15))), 2)

def action_6r(scores: Dict, t: Dict) -> str:
    c = composite(scores); risk = scores.get("risk_score",5)
    dep = (t.get("deployment") or "").lower()
    if bool(t.get("end_of_life")) or c < 2.5: return "Retire"
    if c < 3.5 and scores.get("adoption_rate",5) < 3: return "Retire"
    if c < 4.5 and scores.get("cost_efficiency",5) < 3: return "Replace"
    if risk >= 8.5 and c < 6: return "Replace"
    if c >= 7.5 and risk <= 4: return "Retain"
    if c >= 6.5 and risk <= 5: return "Retain"
    if c >= 6 and "on-prem" in dep and scores.get("technical_health",5) >= 5: return "Rehost"
    if c >= 5.5 and scores.get("technical_health",5) < 5.5: return "Replatform"
    if c >= 4 and scores.get("technical_health",5) < 4.5: return "Refactor"
    return "Retain" if c >= 6 else "Replatform"

def confidence(t: Dict) -> str:
    k = sum(1 for f in ["annual_cost","user_count","criticality","vendor","integrations","age_years","deployment"] if t.get(f))
    return "High" if k >= 5 else "Medium" if k >= 3 else "Low"

def apply_scores(tools: List[Dict]) -> List[Dict]:
    for t in tools:
        sc = score_tool(t)
        t.update(scores=sc, composite_score=composite(sc),
                 rationalization_action=action_6r(sc,t), confidence_level=confidence(t))
    return tools

# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def detect_dups(tools: List[Dict]) -> List[Dict]:
    by_cat: Dict[str,List] = defaultdict(list)
    for t in tools: by_cat[t.get("category","Other")].append(t)
    dups = []
    for cat, grp in by_cat.items():
        for i in range(len(grp)):
            for j in range(i+1, len(grp)):
                a,b = grp[i],grp[j]
                ov = _ov_score(a,b)
                if ov >= 0.45: dups.append(_dup_rec(a,b,ov,cat))
    return sorted(dups, key=lambda x: x["overlap_percentage"], reverse=True)

def _ov_score(a,b):
    s = 0.35
    if (a.get("subcategory") or "") == (b.get("subcategory") or "") and a.get("subcategory"): s += 0.25
    if (a.get("business_unit") or "").lower() == (b.get("business_unit") or "").lower() and a.get("business_unit"): s += 0.15
    va,vb = (a.get("vendor") or "").lower(),(b.get("vendor") or "").lower()
    if va and vb and va != vb: s += 0.10
    ua,ub = a.get("user_count") or 0, b.get("user_count") or 0
    if ua and ub and min(ua,ub)/max(ua,ub) > 0.5: s += 0.10
    return min(1.0, s)

def _dup_rec(a,b,ov,cat):
    pct = round(ov*100)
    sa,sb = a.get("composite_score",5), b.get("composite_score",5)
    ret,con = (a,b) if sa>=sb else (b,a)
    ca,cb = a.get("annual_cost",0) or 0, b.get("annual_cost",0) or 0
    savings = round(min(ca,cb)*(ov/2))
    return {"id":f"dup-{a['id'][:8]}-{b['id'][:8]}","category":cat,
            "tool_a":a.get("name","?"),"tool_b":b.get("name","?"),"overlap_percentage":pct,
            "retain_candidate":ret.get("name"),"consolidate_candidate":con.get("name"),
            "potential_annual_savings":savings,
            "priority":"High" if pct>=70 else "Medium" if pct>=55 else "Low",
            "rationale":(f"{a.get('name')} and {b.get('name')} show {pct}% functional overlap in {cat}. "
                         f"{ret.get('name')} is the strategic retention candidate "
                         f"(score {max(sa,sb):.1f} vs {min(sa,sb):.1f}). "
                         f"Consolidation could yield ~${savings:,}/yr savings.")}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION (in-memory)
# ═══════════════════════════════════════════════════════════════════════════════
CAT_KW = {
    "Monitoring":["monitor","prometheus","grafana","nagios","zabbix","pagerduty"],
    "Logging":["log","elk","elasticsearch","logstash","kibana","fluentd","splunk","sumo logic"],
    "APM":["apm","trace","dynatrace","newrelic","appdynamics","datadog","jaeger"],
    "Security":["security","siem","firewall","iam","pam","crowdstrike","qualys","veracode","snyk"],
    "ITSM":["itsm","incident","ticket","servicenow","jira service","remedy"],
    "Collaboration":["collab","slack","teams","confluence","sharepoint","zoom","webex"],
    "CRM":["crm","salesforce","hubspot","siebel","dynamics crm"],
    "ERP":["erp","sap","workday","peoplesoft"],
    "BSS":["bss","billing","rating","charging"],
    "OSS":["oss","provisioning","network inventory"],
    "Network":["network","sdwan","cisco","juniper","dns","load balanc"],
    "Cloud":["cloud","aws","azure","gcp","terraform","kubernetes","k8s","openstack"],
    "Analytics":["analytic","bi","tableau","powerbi","qlik","looker","data warehouse"],
    "DevOps":["devops","ci/cd","jenkins","gitlab","github actions","sonar","pipeline"],
    "Storage":["storage","backup","netapp","commvault","s3"],
    "Database":["database","postgres","mysql","mongodb","redis","oracle db","sql server"],
}
COL_ALIAS = {
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

def _norm_cat(raw:str)->str:
    low=raw.lower()
    for c,kws in CAT_KW.items():
        if any(k in low for k in kws): return c
    return raw.strip() or "Other"

def _norm_dep(raw:str)->Optional[str]:
    low=raw.lower()
    if any(x in low for x in ["cloud","saas","paas","iaas"]): return "Cloud"
    if any(x in low for x in ["on-prem","onprem","premise"]): return "On-Prem"
    if "hybrid" in low: return "Hybrid"
    return raw or None

def _norm_crit(v)->Optional[str]:
    if not v: return None
    x=str(v).lower()
    if "critical" in x: return "Critical"
    if "high" in x: return "High"
    if "med" in x: return "Medium"
    if "low" in x: return "Low"
    return str(v)

def _s(v:Any)->str:
    return "" if v is None else ("" if str(v).lower() in ("nan","none","null","n/a") else str(v).strip())

def _n(v:Any)->Optional[float]:
    try: return float(str(v).replace(",","").replace("$","").replace("£","").replace("€","").strip())
    except: return None

def _ni(v:Any)->Optional[int]:
    x=_n(v); return int(x) if x is not None else None

def _b(v:Any)->bool:
    return str(v).lower().strip() in ("true","yes","1","y","eol","deprecated","end of life")

def normalize(d:Dict)->Dict:
    t={"id":str(uuid.uuid4()),"name":_s(d.get("name") or d.get("tool_name") or "Unknown Tool"),
       "vendor":_s(d.get("vendor")) or None,"category":_norm_cat(_s(d.get("category",""))),
       "description":_s(d.get("description")) or None,"owner":_s(d.get("owner")) or None,
       "business_unit":_s(d.get("business_unit")) or None,"annual_cost":_n(d.get("annual_cost")),
       "user_count":_ni(d.get("user_count")),"license_type":_s(d.get("license_type")) or None,
       "deployment":_norm_dep(_s(d.get("deployment",""))),
       "criticality":_norm_crit(d.get("criticality")),
       "integrations":_ni(d.get("integrations")),"age_years":_n(d.get("age_years")),
       "end_of_life":_b(d.get("end_of_life",False)),"compliance_required":_b(d.get("compliance_required",False))}
    return {k:v for k,v in t.items() if v not in (None,"") or k in ("id","name","category","end_of_life")}

def parse_csv_bytes(b:bytes)->List[Dict]:
    import pandas as pd
    df=pd.read_csv(io.BytesIO(b))
    return _df(df)

def parse_excel_bytes(b:bytes)->List[Dict]:
    import pandas as pd
    df=pd.read_excel(io.BytesIO(b))
    return _df(df)

def parse_json_bytes(b:bytes)->List[Dict]:
    data=json.loads(b)
    if isinstance(data,list): return [normalize(t) for t in data]
    for v in data.values():
        if isinstance(v,list): return [normalize(t) for t in v]
    return []

def parse_pdf_bytes(b:bytes)->str:
    try:
        import pdfplumber
        txt=""
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for pg in pdf.pages: txt+=pg.extract_text() or ""
        return txt[:8000]
    except: return ""

def _df(df)->List[Dict]:
    import pandas as pd
    df.columns=[c.lower().strip().replace(" ","_").replace("-","_") for c in df.columns]
    df=df.rename(columns={k:v for k,v in COL_ALIAS.items() if k in df.columns})
    df=df.where(pd.notna(df),None)
    return [normalize(r.to_dict()) for _,r in df.iterrows()]

# ═══════════════════════════════════════════════════════════════════════════════
# AI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_json(text:str):
    t=text.strip()
    if t.startswith("```"):
        t=t.split("\n",1)[1] if "\n" in t else t[3:]
        t=t.rsplit("```",1)[0].strip()
    return json.loads(t)

async def ai_parse_text(text:str)->List[Dict]:
    resp=await client.messages.create(
        model=ANTHROPIC_MODEL,temperature=0.1,max_tokens=3000,
        messages=[{"role":"user","content":
            f"""Extract all tools/apps/platforms from the text. Return ONLY valid JSON with a "tools" array — no other text or markdown.
Each item: name, vendor(or null), category(Monitoring/Logging/APM/Security/ITSM/Collaboration/
CRM/ERP/BSS/OSS/Cloud/Analytics/DevOps/Network/Storage/Database/Other),
description(brief/null), annual_cost(number/null), user_count(number/null),
criticality(Critical/High/Medium/Low/null), deployment(Cloud/On-Prem/Hybrid/null),
integrations(number/null), age_years(number/null), end_of_life(bool), compliance_required(bool),
business_unit(string/null).
TEXT:\n{text}"""}])
    try:
        data=_extract_json(resp.content[0].text)
        if isinstance(data,list): return data
        for v in data.values():
            if isinstance(v,list): return v
        return []
    except: return []

async def ai_assess(tools:List[Dict],dups:List[Dict],industry:str,focus:str)->Dict:
    resp=await client.messages.create(
        model=ANTHROPIC_MODEL,temperature=0.2,max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":f"""Perform a comprehensive rationalization assessment.
Industry: {industry.upper()}{(' | Focus: '+focus) if focus else ''}

TOOL INVENTORY (pre-scored, first 30):
{json.dumps(tools[:30],indent=2)[:8000]}

DUPLICATION ANALYSIS:
{json.dumps(dups[:10],indent=2)[:2000]}

Return ONLY valid JSON (no markdown, no explanation) with these exact keys:
{{
  "executive_summary": "<3-5 paragraph CIO-ready narrative>",
  "portfolio_overview": {{"total_tools":<int>,"total_annual_cost":<float>,"portfolio_health":"Healthy|At Risk|Critical","health_rationale":"<brief>"}},
  "rationalization_summary": {{"Retain":<int>,"Rehost":<int>,"Replatform":<int>,"Refactor":<int>,"Replace":<int>,"Retire":<int>}},
  "top_recommendations": [{{"rank":1,"title":"<title>","description":"<detail>","impact":"<impact>","effort":"Low|Medium|High","priority":"Critical|High|Medium","confidence":"High|Medium|Low","timeline":"0-3 months|3-12 months|12-24 months"}}],
  "consolidation_opportunities": [{{"tools":["A","B"],"category":"<cat>","overlap_pct":<int>,"recommended_action":"<action>","estimated_savings":<float>,"rationale":"<why>"}}],
  "risk_highlights": [{{"risk_type":"Security|Compliance|Vendor|Obsolescence|Operational","severity":"Critical|High|Medium","affected_tools":["<names>"],"description":"<desc>","mitigation":"<action>"}}],
  "roadmap": {{"short_term":["<0-3m action>"],"medium_term":["<3-12m action>"],"long_term":["<12-24m action>"]}},
  "expected_outcomes": {{"cost_savings_annual":<float>,"risk_reduction":"<desc>","tool_reduction":"<from X to Y>","strategic_gains":"<value>"}}
}}"""}])
    try: return _extract_json(resp.content[0].text)
    except: return {"executive_summary":resp.content[0].text}

def build_report(tools:List[Dict],dups:List[Dict],assessment:Dict)->str:
    from datetime import datetime
    total_cost=sum(t.get("annual_cost",0) or 0 for t in tools)
    pot_save=sum(d.get("potential_annual_savings",0) or 0 for d in dups)
    cnts:Dict[str,int]={}
    for t in tools:
        a=t.get("rationalization_action","TBD"); cnts[a]=cnts.get(a,0)+1

    BS={"Retain":"color:#155724;background:#d4edda","Rehost":"color:#004085;background:#cce5ff",
        "Replatform":"color:#856404;background:#fff3cd","Refactor":"color:#7d3c00;background:#fde8d8",
        "Replace":"color:#721c24;background:#f8d7da","Retire":"color:#383d41;background:#e2e3e5",
        "High":"color:#721c24;background:#f8d7da","Medium":"color:#856404;background:#fff3cd",
        "Low":"color:#155724;background:#d4edda","Critical":"color:#fff;background:#c0392b"}
    def bdg(a): return f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;{BS.get(a,"color:#333;background:#eee")}">{a}</span>'

    tool_rows="".join(f"<tr><td><b>{t.get('name','')}</b></td><td>{t.get('vendor') or '—'}</td>"
        f"<td>{t.get('category','')}</td><td>{'${:,.0f}'.format(t.get('annual_cost',0)) if t.get('annual_cost') else '—'}</td>"
        f"<td>{t.get('user_count') or '—'}</td><td><b>{t.get('composite_score','—')}</b>/10</td>"
        f"<td>{t.get('scores',{}).get('risk_score','—')}/10</td><td>{bdg(t.get('rationalization_action','TBD'))}</td></tr>" for t in tools)

    dup_rows="".join(f"<tr><td>{d.get('category')}</td><td><b>{d.get('tool_a')}</b></td><td><b>{d.get('tool_b')}</b></td>"
        f"<td><b>{d.get('overlap_percentage')}%</b></td><td>{d.get('retain_candidate')}</td>"
        f"<td>${d.get('potential_annual_savings',0):,.0f}</td><td>{bdg(d.get('priority','Low'))}</td></tr>" for d in dups[:15])

    act_rows="".join(f"<tr><td>{bdg(a)}</td><td><b>{c}</b></td><td>{round(c/max(sum(cnts.values()),1)*100)}%</td></tr>"
        for a,c in cnts.items())

    ex=assessment.get("executive_summary","Run an AI Assessment to generate the executive summary.")
    recs=assessment.get("top_recommendations",[])
    rm=assessment.get("roadmap",{})
    oc=assessment.get("expected_outcomes",{})
    risks=assessment.get("risk_highlights",[])

    rec_html="".join(f"""<div style="background:#f8fbff;border:1px solid #e0e8f8;border-left:4px solid #0063DC;border-radius:8px;padding:16px;margin-bottom:10px">
<div style="display:flex;align-items:center;gap:9px;margin-bottom:7px">
<div style="width:24px;height:24px;background:#0063DC;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700">{r.get('rank','')}</div>
<strong style="color:#003366;flex:1">{r.get('title','')}</strong>
{bdg(r.get('priority','Medium'))} {bdg(r.get('confidence','Medium'))}</div>
<p style="font-size:13px;color:#444;line-height:1.6;margin-bottom:5px">{r.get('description','')}</p>
<p style="font-size:12px;color:#0063DC;font-style:italic">Impact: {r.get('impact','')} · {r.get('timeline','')}</p></div>""" for r in recs[:5])

    def ph_items(key):
        items=rm.get(key,[])
        if isinstance(items,str): items=[items]
        return "".join(f'<li style="padding:6px 0;border-bottom:1px solid rgba(0,0,0,.06)">{x}</li>' for x in items)

    risk_html="".join(f"""<div style="display:flex;gap:12px;background:#fff5f5;border:1px solid #fad7d7;border-radius:9px;padding:14px;margin-bottom:9px">
<div style="font-size:18px">{'🔴' if r.get('severity')=='Critical' else '🟠' if r.get('severity')=='High' else '🟡'}</div>
<div><div style="font-weight:700;color:#1a2340">{r.get('risk_type')} — {r.get('severity')}</div>
<div style="font-size:13px;color:#666;margin:3px 0">{r.get('description','')}</div>
<div style="font-size:12px;color:#00A651">Mitigation: {r.get('mitigation','')}</div></div></div>""" for r in risks)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Tech Rationalization Report — {datetime.now().strftime('%B %Y')}</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4fa;color:#1a2340;font-size:14px}}
.w{{max-width:1200px;margin:0 auto;background:#fff;box-shadow:0 0 40px rgba(0,0,0,.1)}}
.hdr{{background:linear-gradient(135deg,#003366,#0063DC);color:#fff;padding:40px}}
.hdr h1{{font-size:24px;font-weight:700}}.hdr p{{opacity:.75;font-size:13px;margin-top:4px}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#e0e7f0}}
.kc{{background:#fff;padding:24px;text-align:center}}.kv{{font-size:28px;font-weight:800;color:#0063DC}}.kl{{font-size:11px;color:#666;text-transform:uppercase;margin-top:4px}}
.sec{{padding:30px 40px;border-bottom:1px solid #eef1f8}}
h2{{font-size:17px;font-weight:700;color:#003366;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #0063DC}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{background:#003366;color:#fff;padding:10px 13px;text-align:left;font-size:12px;font-weight:600}}
tbody td{{padding:9px 13px;border-bottom:1px solid #f0f3fa}}tbody tr:hover{{background:#f8fbff}}
.exec{{background:#f8fbff;border-left:4px solid #0063DC;padding:20px;border-radius:0 8px 8px 0;font-size:13px;line-height:1.8;white-space:pre-wrap}}
.rm{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.ph{{background:#f8fbff;border-radius:8px;padding:18px}}.ph ul{{list-style:none}}
.ftr{{background:#1a2340;color:#8899bb;padding:18px 40px;display:flex;justify-content:space-between;font-size:12px}}
@media print{{body{{background:#fff}}.w{{box-shadow:none;max-width:100%}}.rm{{grid-template-columns:repeat(3,1fr)}}}}</style></head>
<body><div class="w">
<div class="hdr"><h1>Technology Rationalization Assessment Report</h1>
<p>Enterprise Platform, Application &amp; Tools Assessment · AI-Powered Advisory · KPMG-Style 6R Framework</p>
<p style="margin-top:10px;opacity:.6;font-size:12px">Generated: {datetime.now().strftime('%d %B %Y %H:%M')} · CONFIDENTIAL</p></div>
<div class="kpi">
<div class="kc"><div class="kv">{len(tools)}</div><div class="kl">Tools Assessed</div></div>
<div class="kc"><div class="kv">${total_cost:,.0f}</div><div class="kl">Annual Spend</div></div>
<div class="kc"><div class="kv">{len(dups)}</div><div class="kl">Overlap Pairs</div></div>
<div class="kc"><div class="kv">${pot_save:,.0f}</div><div class="kl">Est. Savings</div></div></div>
<div class="sec"><h2>Executive Summary</h2><div class="exec">{ex}</div></div>
<div class="sec"><h2>Rationalization Action Summary</h2>
<table><thead><tr><th>6R Action</th><th>Count</th><th>% of Portfolio</th></tr></thead>
<tbody>{act_rows}</tbody></table></div>
{'<div class="sec"><h2>Top Priority Recommendations</h2>'+rec_html+'</div>' if rec_html else ''}
{f'''<div class="sec"><h2>Rationalization Roadmap</h2><div class="rm">
<div class="ph" style="border-top:3px solid #00A651"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#00A651;margin-bottom:10px">Phase 1 — Quick Wins (0–3 Months)</p><ul>{ph_items('short_term')}</ul></div>
<div class="ph" style="border-top:3px solid #0063DC"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#0063DC;margin-bottom:10px">Phase 2 — Strategic (3–12 Months)</p><ul>{ph_items('medium_term')}</ul></div>
<div class="ph" style="border-top:3px solid #7B2FBE"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#7B2FBE;margin-bottom:10px">Phase 3 — Transformation (12–24 Months)</p><ul>{ph_items('long_term')}</ul></div>
</div></div>''' if rm else ''}
{f'<div class="sec"><h2>Risk Highlights</h2>'+risk_html+'</div>' if risk_html else ''}
<div class="sec"><h2>Full Tool Portfolio Assessment</h2>
<table><thead><tr><th>Tool</th><th>Vendor</th><th>Category</th><th>Annual Cost</th><th>Users</th><th>Score</th><th>Risk</th><th>Action</th></tr></thead>
<tbody>{tool_rows}</tbody></table></div>
{f'''<div class="sec"><h2>Duplication &amp; Consolidation Opportunities</h2>
<table><thead><tr><th>Category</th><th>Tool A</th><th>Tool B</th><th>Overlap</th><th>Retain</th><th>Est. Savings</th><th>Priority</th></tr></thead>
<tbody>{dup_rows}</tbody></table></div>''' if dups else ''}
{f'''<div class="sec"><h2>Expected Business Outcomes</h2>
<table><tbody>{"".join(f"<tr><td><b>{k.replace('_',' ').title()}</b></td><td>{v}</td></tr>" for k,v in oc.items())}</tbody></table></div>''' if oc else ''}
<div class="ftr"><span>Tech Rationalization AI Agent · Enterprise Technology Strategy Advisory</span>
<span>CONFIDENTIAL · {datetime.now().strftime('%Y')}</span></div></div></body></html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Tech Rationalization AI Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

@app.get("/", response_class=HTMLResponse)
async def ui(): return HTMLResponse(get_html())

@app.post("/api/chat")
async def chat(req: ChatReq):
    system_parts = [SYSTEM_PROMPT]
    if req.tools:
        summary = "\n".join(f"- {t.get('name')} | {t.get('category')} | Action:{t.get('rationalization_action','TBD')} | Score:{t.get('composite_score','?')}" for t in req.tools[:30])
        system_parts.append(f"Current tool inventory ({len(req.tools)} tools):\n{summary}")
    msgs = [{"role":m.get("role","user"),"content":m.get("content","")}
            for m in req.history[-20:] if m.get("role") in ("user","assistant")]
    msgs.append({"role":"user","content":req.message})
    resp = await client.messages.create(
        model=ANTHROPIC_MODEL, system="\n\n".join(system_parts), messages=msgs,
        max_tokens=2000, temperature=0.2)
    return {"reply": resp.content[0].text}

@app.post("/api/ingest")
async def ingest(file: Optional[UploadFile]=File(None), text: Optional[str]=Form(None)):
    tools: List[Dict] = []
    if file and file.filename:
        content = await file.read()
        ext = file.filename.lower().rsplit(".",1)[-1]
        if   ext == "csv":           tools = parse_csv_bytes(content)
        elif ext in ("xlsx","xls"):  tools = parse_excel_bytes(content)
        elif ext == "json":          tools = parse_json_bytes(content)
        elif ext == "pdf":
            raw = parse_pdf_bytes(content)
            if raw: tools = [normalize(t) for t in await ai_parse_text(raw)]
    elif text:
        tools = [normalize(t) for t in await ai_parse_text(text)]
    tools = apply_scores(tools)
    dups  = detect_dups(tools)
    return {"tools":tools,"duplications":dups,
            "summary":{"total_tools":len(tools),
                        "total_annual_cost":sum(t.get("annual_cost",0) or 0 for t in tools),
                        "duplications_found":len(dups),
                        "potential_savings":sum(d.get("potential_annual_savings",0) for d in dups)}}

@app.post("/api/assess")
async def assess(req: AssessReq):
    if not req.tools: raise HTTPException(400,"No tools provided.")
    result = await ai_assess(req.tools, req.duplications, req.industry, req.focus or "")
    return {"assessment": result}

@app.post("/api/report")
async def report(req: ReportReq):
    return {"html": build_report(req.tools, req.duplications, req.assessment)}

@app.get("/api/health")
async def health(): return {"status":"healthy","model":ANTHROPIC_MODEL,"key_set":bool(ANTHROPIC_API_KEY)}

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED HTML (complete single-page application)
# ═══════════════════════════════════════════════════════════════════════════════
def get_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tech Rationalization AI Agent</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--navy:#003366;--blue:#0063DC;--sky:#00A3E0;--green:#00A651;--amber:#FFC200;--red:#E31837;--purple:#7B2FBE;--bg:#F0F4FA;--sur:#fff;--bdr:#E0E7F0;--txt:#1A2340;--mut:#6B7A99}
html,body{height:100%;font-family:'Segoe UI',system-ui,Arial,sans-serif;background:var(--bg);color:var(--txt);font-size:14px}
.app{display:flex;height:100vh;overflow:hidden}
/* Sidebar */
.sb{width:230px;background:var(--navy);display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
.sb-logo{display:flex;align-items:center;gap:10px;padding:18px 16px;border-bottom:1px solid rgba(255,255,255,.1)}
.sb-logo .ic{width:36px;height:36px;background:linear-gradient(135deg,var(--blue),var(--sky));border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:900;color:#fff;flex-shrink:0}
.sb-logo strong{color:#fff;font-size:13px;display:block;line-height:1.2}.sb-logo span{color:rgba(255,255,255,.45);font-size:11px}
.sb-sec{padding:14px 10px 4px}
.sb-lbl{font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,.35);padding:0 8px;margin-bottom:5px}
.nav{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:7px;cursor:pointer;color:rgba(255,255,255,.65);font-size:13px;font-weight:500;transition:all .15s;margin-bottom:2px}
.nav:hover{background:rgba(255,255,255,.08);color:#fff}.nav.on{background:var(--blue);color:#fff}
.ni{font-size:15px;width:20px;text-align:center;flex-shrink:0}
.nb{margin-left:auto;background:var(--red);color:#fff;border-radius:10px;font-size:10px;font-weight:700;padding:2px 6px;display:none}.nb.show{display:inline-block}
.sb-ft{margin-top:auto;padding:14px;border-top:1px solid rgba(255,255,255,.1)}
.sc{background:rgba(255,255,255,.07);border-radius:7px;padding:10px}
.sl{font-size:10px;color:rgba(255,255,255,.4);text-transform:uppercase;letter-spacing:.7px}
.sv{font-size:11px;color:rgba(255,255,255,.65);font-family:monospace;margin-top:3px}
/* Main */
.main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}
.topbar{height:60px;background:var(--sur);border-bottom:1px solid var(--bdr);display:flex;align-items:center;padding:0 24px;gap:14px;flex-shrink:0}
.tt{font-size:17px;font-weight:700;color:var(--navy);flex:1}.ts{font-size:11px;color:var(--mut)}
.cnt{flex:1;overflow-y:auto;padding:24px}
/* Tabs */
.tab{display:none}.tab.on{display:block}
/* KPI */
.kg{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.kc{background:var(--sur);border-radius:10px;border:1px solid var(--bdr);padding:18px 20px;display:flex;align-items:center;gap:14px}
.ki{width:44px;height:44px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.kv{font-size:26px;font-weight:800;color:var(--navy);line-height:1}.kl{font-size:11px;color:var(--mut);margin-top:3px}
/* Cards */
.card{background:var(--sur);border-radius:10px;border:1px solid var(--bdr);padding:20px;margin-bottom:16px}
.ch{display:flex;align-items:center;gap:8px;margin-bottom:16px}.ct{font-size:14px;font-weight:700;color:var(--navy)}
/* Charts */
.cg{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.cw{height:220px;position:relative}
/* Badges */
.b{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap}
.br{color:#155724;background:#d4edda}.bh{color:#004085;background:#cce5ff}
.bp{color:#856404;background:#fff3cd}.bf{color:#7d3c00;background:#fde8d8}
.bl{color:#721c24;background:#f8d7da}.bt{color:#383d41;background:#e2e3e5}
.bhi{color:#721c24;background:#f8d7da}.bme{color:#856404;background:#fff3cd}
.blo{color:#155724;background:#d4edda}.bcr{color:#fff;background:#c0392b}
/* Btns */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:all .15s}
.bp1{background:var(--blue);color:#fff}.bp1:hover{background:#004eb3}
.bs1{background:var(--bg);color:var(--txt);border:1px solid var(--bdr)}.bs1:hover{background:var(--bdr)}
.bsm{padding:6px 12px;font-size:12px}.btn:disabled{opacity:.5;cursor:not-allowed}
/* Table */
.tw{overflow-x:auto;border-radius:8px;border:1px solid var(--bdr)}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{background:var(--navy);color:#fff;padding:10px 13px;text-align:left;font-size:11px;font-weight:700;white-space:nowrap}
tbody td{padding:9px 13px;border-bottom:1px solid var(--bdr);vertical-align:middle}
tbody tr:last-child td{border-bottom:none}tbody tr:hover{background:#f8fbff}
/* Score bar */
.sb2{display:flex;align-items:center;gap:7px;min-width:90px}
.bt2{flex:1;height:5px;background:var(--bdr);border-radius:3px;overflow:hidden}
.bf2{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--blue),var(--sky))}
.bl2{font-size:11px;font-weight:700;min-width:24px;text-align:right}
.sh{color:var(--green)}.sm2{color:var(--amber)}.sl2{color:var(--red)}
/* Upload */
.uz{border:2px dashed var(--bdr);border-radius:10px;padding:40px;text-align:center;cursor:pointer;transition:all .2s;background:#fafbff}
.uz:hover,.uz.dg{border-color:var(--blue);background:#eef4ff}
.ui2{font-size:40px;margin-bottom:10px}
/* Chat */
.cl{display:flex;flex-direction:column;height:calc(100vh - 108px)}
.cm{flex:1;overflow-y:auto;padding:12px 0;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;max-width:88%}
.mu{align-self:flex-end;flex-direction:row-reverse}
.mav{width:34px;height:34px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700}
.mu .mav{background:var(--blue);color:#fff}.ma .mav{background:var(--navy);color:#fff}
.mb2{padding:11px 15px;border-radius:10px;font-size:13px;line-height:1.7;max-width:100%}
.mu .mb2{background:var(--blue);color:#fff;border-radius:10px 2px 10px 10px}
.ma .mb2{background:var(--sur);border:1px solid var(--bdr);border-radius:2px 10px 10px 10px;white-space:pre-wrap}
.ma .mb2 strong{color:var(--navy)}
.cib{padding:12px 0 0;display:flex;gap:8px}
.ci{flex:1;padding:11px 14px;border-radius:9px;border:1px solid var(--bdr);font-size:13px;resize:none;font-family:inherit;line-height:1.5;max-height:140px}
.ci:focus{outline:none;border-color:var(--blue)}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}
.chip{background:#eef4ff;border:1px solid #c7d8ff;border-radius:20px;padding:5px 13px;font-size:12px;color:var(--blue);cursor:pointer;transition:all .15s}
.chip:hover{background:var(--blue);color:#fff;border-color:var(--blue)}
/* Assess */
.rc{background:var(--sur);border:1px solid var(--bdr);border-left:4px solid var(--blue);border-radius:9px;padding:16px;margin-bottom:11px}
.rh{display:flex;align-items:center;gap:9px;margin-bottom:7px}
.rr{width:24px;height:24px;background:var(--blue);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}
.rmc{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.ph{background:var(--sur);border-radius:9px;padding:18px}
.ph.p1{border-top:3px solid var(--green)}.ph.p2{border-top:3px solid var(--blue)}.ph.p3{border-top:3px solid var(--purple)}
.pht{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.p1 .pht{color:var(--green)}.p2 .pht{color:var(--blue)}.p3 .pht{color:var(--purple)}
.pi{display:flex;gap:7px;align-items:flex-start;font-size:13px;padding:7px 0;border-bottom:1px solid var(--bdr);line-height:1.5}
.pi:last-child{border-bottom:none}
.pd{width:16px;height:16px;border-radius:50%;flex-shrink:0;margin-top:2px}
.p1 .pd{background:#d4edda}.p2 .pd{background:#cce5ff}.p3 .pd{background:#e8d8f8}
.eb{background:#f8fbff;border-left:4px solid var(--blue);border-radius:0 9px 9px 0;padding:18px;font-size:13px;line-height:1.8;white-space:pre-wrap}
.dc{display:flex;align-items:flex-start;gap:12px;background:#fff8f0;border:1px solid #fde8c8;border-radius:9px;padding:14px;margin-bottom:9px}
.dp{font-size:18px;font-weight:800;color:var(--amber);min-width:46px;text-align:center;flex-shrink:0}
/* Filters */
.fb{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
.fl{padding:7px 12px;border-radius:7px;border:1px solid var(--bdr);font-size:13px;background:var(--sur);color:var(--txt)}
.fl:focus{outline:none;border-color:var(--blue)}
/* Loading */
.ld{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:1000;align-items:center;justify-content:center;flex-direction:column;color:#fff;gap:14px;font-size:14px}
.ld.show{display:flex}
.sp{width:44px;height:44px;border:4px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .65s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* Toast */
.ts2{position:fixed;bottom:22px;right:22px;display:flex;flex-direction:column;gap:9px;z-index:2000}
.tt2{background:var(--navy);color:#fff;padding:11px 16px;border-radius:9px;font-size:13px;box-shadow:0 4px 20px rgba(0,0,0,.25);display:flex;align-items:center;gap:9px;min-width:260px;animation:si .22s ease}
.ts2 .ok{border-left:4px solid var(--green)}.ts2 .er{border-left:4px solid var(--red)}.ts2 .in{border-left:4px solid var(--blue)}
@keyframes si{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}
/* Misc */
.es{text-align:center;padding:56px 24px;color:var(--mut)}
.ei{font-size:50px;margin-bottom:14px}
.es h3{font-size:17px;font-weight:700;color:var(--navy);margin-bottom:7px}
.fx{display:flex}.ic2{align-items:center}.g2{gap:8px}.jb{justify-content:space-between}
.m4{margin-bottom:16px}.mt4{margin-top:16px}.wf{width:100%}.tx{font-size:12px}.mu2{color:var(--mut)}
textarea{padding:12px;border:1px solid var(--bdr);border-radius:8px;font-size:13px;font-family:inherit;line-height:1.6;width:100%}
textarea:focus{outline:none;border-color:var(--blue)}
select,input[type=text]{padding:7px 12px;border-radius:7px;border:1px solid var(--bdr);font-size:13px;background:var(--sur);color:var(--txt)}
select:focus,input:focus{outline:none;border-color:var(--blue)}
@media(max-width:900px){.kg{grid-template-columns:1fr 1fr}.cg{grid-template-columns:1fr}.rmc{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="app">
<!-- SIDEBAR -->
<aside class="sb">
  <div class="sb-logo">
    <div class="ic">&#9889;</div>
    <div><strong>TechRatio AI</strong><span>Enterprise Advisory</span></div>
  </div>
  <div class="sb-sec">
    <div class="sb-lbl">Main</div>
    <div class="nav on" data-tab="dashboard"><span class="ni">&#128202;</span><span>Dashboard</span></div>
    <div class="nav" data-tab="ingest"><span class="ni">&#128229;</span><span>Data Ingest</span></div>
  </div>
  <div class="sb-sec">
    <div class="sb-lbl">Analysis</div>
    <div class="nav" data-tab="inventory"><span class="ni">&#128193;</span><span>Tool Inventory</span><span class="nb" id="nb">0</span></div>
    <div class="nav" data-tab="assessment"><span class="ni">&#127919;</span><span>AI Assessment</span></div>
    <div class="nav" data-tab="chat"><span class="ni">&#128172;</span><span>AI Advisor</span></div>
  </div>
  <div class="sb-sec">
    <div class="sb-lbl">Output</div>
    <div class="nav" data-tab="reports"><span class="ni">&#128196;</span><span>Reports</span></div>
  </div>
  <div class="sb-ft">
    <div class="sc"><div class="sl">Model</div><div class="sv" id="mdl">gpt-4o</div></div>
  </div>
</aside>
<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div><div class="tt" id="pt">Dashboard</div><div class="ts">KPMG-Style 6R Rationalization &bull; Telecom &amp; Enterprise AI Agent</div></div>
    <div class="fx g2">
      <button class="btn bs1 bsm" onclick="go('ingest')">+ Ingest Data</button>
      <button class="btn bp1 bsm" onclick="go('assessment');runAssess()">&#127919; Run Assessment</button>
    </div>
  </div>
  <div class="cnt">

  <!-- DASHBOARD -->
  <div class="tab on" id="tab-dashboard">
    <div class="kg">
      <div class="kc"><div class="ki" style="background:#eef4ff">&#128230;</div><div><div class="kv" id="k0">0</div><div class="kl">Tools Assessed</div></div></div>
      <div class="kc"><div class="ki" style="background:#fff3e0">&#128176;</div><div><div class="kv" id="k1">$0</div><div class="kl">Annual Spend</div></div></div>
      <div class="kc"><div class="ki" style="background:#ffeaea">&#128257;</div><div><div class="kv" id="k2">0</div><div class="kl">Overlap Pairs</div></div></div>
      <div class="kc"><div class="ki" style="background:#e8f8ef">&#128161;</div><div><div class="kv" id="k3">$0</div><div class="kl">Est. Savings</div></div></div>
    </div>
    <div class="cg">
      <div class="card"><div class="ch"><span>&#127919;</span><span class="ct">6R Rationalization Distribution</span></div><div class="cw"><canvas id="cd"></canvas></div></div>
      <div class="card"><div class="ch"><span>&#128200;</span><span class="ct">Top Tools by Score</span></div><div class="cw"><canvas id="cb"></canvas></div></div>
    </div>
    <div style="display:grid;grid-template-columns:1.5fr 1fr;gap:14px">
      <div class="card">
        <div class="ch"><span>&#128193;</span><span class="ct">Recent Assessments</span></div>
        <div class="tw"><table><thead><tr><th>Tool</th><th>Category</th><th>Score</th><th>Action</th></tr></thead>
        <tbody id="dt"><tr><td colspan="4" style="text-align:center;padding:32px;color:var(--mut)">No data &mdash; ingest your portfolio</td></tr></tbody></table></div>
      </div>
      <div class="card">
        <div class="ch"><span>&#9888;&#65039;</span><span class="ct">Duplication Alerts</span></div>
        <div id="dd"><p class="mu2 tx">Ingest tools to detect duplications.</p></div>
      </div>
    </div>
    <div class="card mt4" style="background:linear-gradient(135deg,#003366,#0063DC);color:#fff;border:none">
      <div style="display:flex;align-items:center;gap:18px">
        <div style="font-size:44px">&#129302;</div>
        <div style="flex:1">
          <div style="font-size:16px;font-weight:700;margin-bottom:5px">Platform, Application &amp; Tools Rationalization AI Agent</div>
          <div style="font-size:12px;opacity:.8;line-height:1.6">Powered by OpenAI GPT-4o &bull; KPMG-Style 6R Model &bull; Telecom &amp; TMT Optimized &bull; Upload your tool inventory &bull; Get AI-powered scores &bull; Generate CIO-ready roadmaps &amp; reports</div>
        </div>
        <div class="fx g2">
          <button class="btn bsm" style="background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3)" onclick="go('chat')">Chat with AI</button>
          <button class="btn bsm" style="background:#00A651;color:#fff" onclick="go('ingest')">Get Started &rarr;</button>
        </div>
      </div>
    </div>
  </div>

  <!-- INGEST -->
  <div class="tab" id="tab-ingest">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
      <div>
        <div class="card">
          <div class="ch"><span>&#128193;</span><span class="ct">Upload File (CSV / Excel / JSON / PDF)</span></div>
          <div class="uz" id="uz">
            <div class="ui2">&#9729;&#65039;</div>
            <h3 style="font-size:15px;font-weight:700;color:var(--navy);margin-bottom:6px">Drop file here or click to upload</h3>
            <p id="un" style="color:var(--mut);font-size:13px">CSV, Excel, JSON, PDF supported</p>
            <div style="display:flex;gap:6px;justify-content:center;margin-top:12px">
              <span style="background:var(--bg);border:1px solid var(--bdr);padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;color:var(--mut)">CSV</span>
              <span style="background:var(--bg);border:1px solid var(--bdr);padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;color:var(--mut)">XLSX</span>
              <span style="background:var(--bg);border:1px solid var(--bdr);padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;color:var(--mut)">JSON</span>
              <span style="background:var(--bg);border:1px solid var(--bdr);padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;color:var(--mut)">PDF</span>
            </div>
            <input type="file" id="fi" accept=".csv,.xlsx,.xls,.json,.pdf" style="display:none">
          </div>
        </div>
        <div class="card">
          <div class="ch"><span>&#9999;&#65039;</span><span class="ct">Or Paste Free Text Description</span></div>
          <textarea id="ti" rows="10" placeholder="Describe tools in plain text. Example:&#10;&#10;Tool: Splunk Enterprise&#10;Vendor: Splunk&#10;Category: Logging&#10;Annual Cost: 180000&#10;Users: 150&#10;Criticality: High&#10;Deployment: On-Prem&#10;Age Years: 6&#10;&#10;Tool: Dynatrace&#10;Vendor: Dynatrace&#10;Category: APM&#10;Annual Cost: 95000&#10;Users: 60&#10;Criticality: High&#10;Deployment: Cloud"></textarea>
          <div class="fx g2 mt4">
            <button class="btn bp1" id="bi">&#9889; Ingest &amp; Score Tools</button>
            <button class="btn bs1" id="bsmp">Load Sample Data</button>
          </div>
        </div>
      </div>
      <div>
        <div class="card m4">
          <div class="ch"><span>&#8505;&#65039;</span><span class="ct">How It Works</span></div>
          <div style="display:flex;flex-direction:column;gap:14px">
            <div class="fx g2 ic2"><div style="width:30px;height:30px;background:#eef4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--blue);flex-shrink:0">1</div><div><strong>Ingest</strong> &mdash; Upload CSV/Excel/JSON/PDF or paste free text tool descriptions</div></div>
            <div class="fx g2 ic2"><div style="width:30px;height:30px;background:#eef4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--blue);flex-shrink:0">2</div><div><strong>Score</strong> &mdash; AI scores each tool on 7 dimensions with 6R action assigned</div></div>
            <div class="fx g2 ic2"><div style="width:30px;height:30px;background:#eef4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--blue);flex-shrink:0">3</div><div><strong>Detect</strong> &mdash; Finds overlapping tools with consolidation savings estimate</div></div>
            <div class="fx g2 ic2"><div style="width:30px;height:30px;background:#eef4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--blue);flex-shrink:0">4</div><div><strong>Assess</strong> &mdash; GPT-4o generates executive recommendations &amp; roadmap</div></div>
            <div class="fx g2 ic2"><div style="width:30px;height:30px;background:#eef4ff;border-radius:7px;display:flex;align-items:center;justify-content:center;font-weight:800;color:var(--blue);flex-shrink:0">5</div><div><strong>Export</strong> &mdash; Download print-ready HTML report for CIO/CTO presentation</div></div>
          </div>
        </div>
        <div class="card">
          <div class="ch"><span>&#127991;&#65039;</span><span class="ct">6R Action Reference</span></div>
          <div style="display:flex;flex-direction:column;gap:9px">
            <div class="fx g2 ic2"><span class="b br">Retain</span><span class="tx">Strategic &amp; healthy &mdash; no immediate action</span></div>
            <div class="fx g2 ic2"><span class="b bh">Rehost</span><span class="tx">Lift-and-shift to cloud infrastructure</span></div>
            <div class="fx g2 ic2"><span class="b bp">Replatform</span><span class="tx">Minor modernization with managed services</span></div>
            <div class="fx g2 ic2"><span class="b bf">Refactor</span><span class="tx">Significant redesign &amp; re-architecture</span></div>
            <div class="fx g2 ic2"><span class="b bl">Replace</span><span class="tx">Better market alternative &mdash; plan migration</span></div>
            <div class="fx g2 ic2"><span class="b bt">Retire</span><span class="tx">Decommission &mdash; redundant or end-of-life</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- INVENTORY -->
  <div class="tab" id="tab-inventory">
    <div class="fb">
      <input type="text" id="is" placeholder="Search name or vendor&hellip;" style="flex:1;max-width:280px">
      <select id="ic3" class="fl"><option value="">All Categories</option></select>
      <select id="ia" class="fl">
        <option value="">All Actions</option>
        <option>Retain</option><option>Rehost</option><option>Replatform</option>
        <option>Refactor</option><option>Replace</option><option>Retire</option>
      </select>
    </div>
    <div class="card m4">
      <div class="ch"><span>&#128193;</span><span class="ct">Tool Portfolio</span></div>
      <div class="tw"><table>
        <thead><tr><th>Tool Name</th><th>Vendor</th><th>Category</th><th>Deployment</th><th>Annual Cost</th><th>Users</th><th>Score</th><th>Risk</th><th>Confidence</th><th>6R Action</th></tr></thead>
        <tbody id="ib"><tr><td colspan="10" style="text-align:center;padding:40px;color:var(--mut)">No tools ingested yet &mdash; go to <strong>Data Ingest</strong></td></tr></tbody>
      </table></div>
    </div>
    <div class="card">
      <div class="ch"><span>&#128257;</span><span class="ct">Duplication &amp; Consolidation Opportunities</span></div>
      <div class="tw"><table>
        <thead><tr><th>Category</th><th>Tool A</th><th>Tool B</th><th>Overlap</th><th>Retain Candidate</th><th>Est. Savings/yr</th><th>Priority</th></tr></thead>
        <tbody id="db2"><tr><td colspan="7" style="text-align:center;padding:28px;color:var(--mut)">No duplications detected.</td></tr></tbody>
      </table></div>
    </div>
  </div>

  <!-- ASSESSMENT -->
  <div class="tab" id="tab-assessment">
    <div class="card m4">
      <div class="ch"><span>&#9881;&#65039;</span><span class="ct">Assessment Configuration</span></div>
      <div style="display:grid;grid-template-columns:1fr 1fr auto;gap:14px;align-items:end">
        <div>
          <label style="display:block;font-size:11px;font-weight:700;margin-bottom:5px;color:var(--mut)">INDUSTRY CONTEXT</label>
          <select id="ai2" style="width:100%">
            <option value="telecom">Telecom / TMT</option>
            <option value="banking">Banking / Financial Services</option>
            <option value="healthcare">Healthcare</option>
            <option value="retail">Retail / E-Commerce</option>
            <option value="enterprise">Large Enterprise</option>
          </select>
        </div>
        <div>
          <label style="display:block;font-size:11px;font-weight:700;margin-bottom:5px;color:var(--mut)">FOCUS AREA (OPTIONAL)</label>
          <input type="text" id="af" placeholder="e.g., Observability stack, Security tools&hellip;" style="width:100%">
        </div>
        <button class="btn bp1" id="ba">&#127919; Run AI Assessment</button>
      </div>
    </div>
    <div id="ar">
      <div class="es">
        <div class="ei">&#129302;</div>
        <h3>No Assessment Yet</h3>
        <p>Ingest your tool inventory, then click <strong>Run AI Assessment</strong> to get GPT-4o powered recommendations.</p>
        <div class="fx g2 mt4" style="justify-content:center">
          <button class="btn bs1" onclick="go('ingest')">Ingest Data First</button>
          <button class="btn bp1" onclick="runAssess()">Run Assessment Now</button>
        </div>
      </div>
    </div>
  </div>

  <!-- CHAT -->
  <div class="tab" id="tab-chat">
    <div class="cl">
      <div class="chips" id="chs">
        <span class="chip">Assess our current application portfolio</span>
        <span class="chip">Which tools should we retire this year?</span>
        <span class="chip">Identify consolidation opportunities</span>
        <span class="chip">Generate executive summary for CIO</span>
        <span class="chip">What are our highest security risks?</span>
        <span class="chip">Create a 12-month rationalization roadmap</span>
      </div>
      <div class="cm" id="cm">
        <div class="msg ma">
          <div class="mav">AI</div>
          <div class="mb2"><strong>Tech Rationalization AI Agent &mdash; Ready</strong><br><br>
I am your enterprise technology strategy consultant, specializing in Platform, Application &amp; Tools rationalization using KPMG-style frameworks optimized for Telecom and large enterprise environments.<br><br>
I can help you:<br>
&bull; <strong>Assess</strong> your tool portfolio across 7 scoring dimensions<br>
&bull; <strong>Apply</strong> the 6R model (Retain / Rehost / Replatform / Refactor / Replace / Retire)<br>
&bull; <strong>Identify</strong> duplication, redundancy, and consolidation opportunities<br>
&bull; <strong>Generate</strong> CIO/CTO-ready executive recommendations and roadmaps<br><br>
Upload your tool inventory in <strong>Data Ingest</strong> for full portfolio analysis, or ask me anything directly.</div>
        </div>
      </div>
      <div class="cib">
        <textarea class="ci" id="cin" rows="2" placeholder="Ask anything &mdash; 'Which tools should we retire?', 'Identify APM stack duplication', 'Create telecom CMP rationalization roadmap'&hellip;"></textarea>
        <button class="btn bp1" id="bsnd" style="align-self:flex-end;height:46px">Send</button>
      </div>
    </div>
  </div>

  <!-- REPORTS -->
  <div class="tab" id="tab-reports">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
      <div>
        <div class="card m4">
          <div class="ch"><span>&#128196;</span><span class="ct">Generate Executive Report</span></div>
          <p class="tx mu2 m4" style="line-height:1.6">Generates a comprehensive HTML report for CIO/CTO presentation. Includes full portfolio assessment, AI recommendations, roadmap, risk analysis, and ROI outcomes.</p>
          <div class="card" style="border:2px solid var(--blue);margin-bottom:0">
            <div class="fx g2 ic2 m4">
              <span style="font-size:22px">&#128203;</span>
              <div><div style="font-weight:700">Executive Summary Report</div><div class="tx mu2">Full assessment &bull; Recommendations &bull; Roadmap &bull; ROI</div></div>
            </div>
            <button class="btn bp1 wf" id="brpt">&#11015; Download HTML Report</button>
          </div>
        </div>
        <div class="card">
          <div class="ch"><span>&#9989;</span><span class="ct">Report Includes</span></div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <div class="fx g2"><span>&#9989;</span><span class="tx">Executive Summary (CIO-ready narrative)</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Portfolio Overview &amp; Health Score</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">6R Rationalization Action Summary</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Top 5 Priority Recommendations with Impact</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Full Tool Assessment Table with Scores</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Duplication &amp; Consolidation Analysis</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">3-Phase Rationalization Roadmap</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Risk Highlights &amp; Mitigations</span></div>
            <div class="fx g2"><span>&#9989;</span><span class="tx">Expected Business Outcomes &amp; ROI</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="ch"><span>&#128065;&#65039;</span><span class="ct">Portfolio Snapshot</span></div>
        <div id="rp"><p class="mu2 tx">Ingest data to see portfolio snapshot.</p></div>
      </div>
    </div>
  </div>

  </div><!-- /cnt -->
</div><!-- /main -->
</div><!-- /app -->

<div class="ld" id="ld"><div class="sp"></div><div id="lm">Processing&hellip;</div></div>
<div class="ts2" id="ts2"></div>

<script>
'use strict';
let TOOLS=[], DUPS=[], HIST=[], ASSESS={};
let cD=null, cB=null;
const A='';

async function init(){
  setupNav(); setupIngest(); setupChat();
  document.getElementById('ba').onclick=runAssess;
  document.getElementById('brpt').onclick=dlReport;
  try{const r=await fetch(A+'/api/health');const d=await r.json();document.getElementById('mdl').textContent=d.model||'gpt-4o';}catch(_){}
}

function setupNav(){
  document.querySelectorAll('.nav').forEach(el=>el.addEventListener('click',()=>go(el.dataset.tab)));
}
function go(tab){
  document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('on',el.dataset.tab===tab));
  document.querySelectorAll('.tab').forEach(el=>el.classList.toggle('on',el.id==='tab-'+tab));
  document.getElementById('pt').textContent={dashboard:'Dashboard',ingest:'Data Ingest',inventory:'Tool Inventory',assessment:'AI Assessment',chat:'AI Advisor',reports:'Reports & Export'}[tab]||tab;
}

function setupIngest(){
  const uz=document.getElementById('uz'), fi=document.getElementById('fi');
  uz.onclick=()=>fi.click();
  uz.ondragover=e=>{e.preventDefault();uz.classList.add('dg');};
  uz.ondragleave=()=>uz.classList.remove('dg');
  uz.ondrop=e=>{e.preventDefault();uz.classList.remove('dg');if(e.dataTransfer.files[0]){fi.files=e.dataTransfer.files;document.getElementById('un').textContent='Selected: '+fi.files[0].name;}};
  fi.onchange=()=>{if(fi.files[0])document.getElementById('un').textContent='Selected: '+fi.files[0].name;};
  document.getElementById('bi').onclick=doIngest;
  document.getElementById('bsmp').onclick=loadSample;
}

async function doIngest(){
  const file=document.getElementById('fi').files[0], text=document.getElementById('ti').value.trim();
  if(!file&&!text)return toast('Please upload a file or enter tool descriptions','er');
  showLd('Ingesting and scoring tools with AI...');
  const form=new FormData();
  if(file)form.append('file',file);
  if(text)form.append('text',text);
  try{
    const r=await fetch(A+'/api/ingest',{method:'POST',body:form});
    const d=await r.json(); hideLd();
    if(!r.ok)return toast(d.detail||'Ingest failed','er');
    TOOLS=d.tools; DUPS=d.duplications;
    renderDash(); renderInv();
    const nb=document.getElementById('nb'); nb.textContent=TOOLS.length; nb.classList.add('show');
    toast('Ingested '+d.summary.total_tools+' tools successfully','ok');
    go('inventory');
  }catch(e){hideLd();toast('Error: '+e.message,'er');}
}

function loadSample(){
  document.getElementById('ti').value=`Tool: Splunk Enterprise
Vendor: Splunk
Category: Logging
Annual Cost: 180000
Users: 150
Criticality: High
Deployment: On-Prem
Integrations: 18
Age Years: 6

Tool: Elastic Stack ELK
Vendor: Elastic
Category: Logging
Annual Cost: 45000
Users: 80
Criticality: Medium
Deployment: Cloud
Integrations: 12
Age Years: 3

Tool: Dynatrace
Vendor: Dynatrace
Category: APM
Annual Cost: 95000
Users: 60
Criticality: High
Deployment: Cloud
Integrations: 22
Age Years: 2

Tool: AppDynamics
Vendor: Cisco
Category: APM
Annual Cost: 88000
Users: 45
Criticality: Medium
Deployment: Cloud
Integrations: 10
Age Years: 5

Tool: Nagios XI
Vendor: Nagios Enterprises
Category: Monitoring
Annual Cost: 12000
Users: 20
Criticality: Medium
Deployment: On-Prem
Integrations: 5
Age Years: 11

Tool: Prometheus Grafana
Vendor: Open Source
Category: Monitoring
Annual Cost: 8000
Users: 40
Criticality: High
Deployment: Cloud
Integrations: 15
Age Years: 3

Tool: ServiceNow ITSM
Vendor: ServiceNow
Category: ITSM
Annual Cost: 220000
Users: 500
Criticality: Critical
Deployment: Cloud
Integrations: 30
Age Years: 4

Tool: Jira Service Management
Vendor: Atlassian
Category: ITSM
Annual Cost: 55000
Users: 200
Criticality: Medium
Deployment: Cloud
Integrations: 15
Age Years: 3

Tool: CrowdStrike Falcon
Vendor: CrowdStrike
Category: Security
Annual Cost: 130000
Users: 1000
Criticality: Critical
Deployment: Cloud
Integrations: 8
Age Years: 2

Tool: Symantec Antivirus Legacy
Vendor: Broadcom
Category: Security
Annual Cost: 65000
Users: 800
Criticality: High
Deployment: On-Prem
Integrations: 2
Age Years: 9
End of Life: true

Tool: Salesforce CRM
Vendor: Salesforce
Category: CRM
Annual Cost: 310000
Users: 300
Criticality: Critical
Deployment: Cloud
Integrations: 25
Age Years: 6

Tool: Siebel CRM
Vendor: Oracle
Category: CRM
Annual Cost: 120000
Users: 150
Criticality: High
Deployment: On-Prem
Integrations: 8
Age Years: 12`;
  toast('Sample data loaded — click Ingest to process','in');
}

function renderDash(){
  const tc=TOOLS.reduce((s,t)=>s+(t.annual_cost||0),0);
  const ps=DUPS.reduce((s,d)=>s+(d.potential_annual_savings||0),0);
  document.getElementById('k0').textContent=TOOLS.length;
  document.getElementById('k1').textContent='$'+fmt(tc);
  document.getElementById('k2').textContent=DUPS.length;
  document.getElementById('k3').textContent='$'+fmt(ps);
  const cnt={};TOOLS.forEach(t=>{const a=t.rationalization_action||'TBD';cnt[a]=(cnt[a]||0)+1;});
  rDonut(cnt); rBar();
  document.getElementById('dt').innerHTML=TOOLS.slice(0,6).map(t=>`<tr>
    <td><strong>${e(t.name)}</strong></td><td>${e(t.category)}</td>
    <td><div class="sb2"><div class="bt2"><div class="bf2" style="width:${(t.composite_score||0)*10}%"></div></div>
    <span class="bl2 ${sc(t.composite_score)}">${t.composite_score||'—'}</span></div></td>
    <td>${bdg(t.rationalization_action||'TBD')}</td></tr>`).join('');
  document.getElementById('dd').innerHTML=DUPS.length
    ?DUPS.slice(0,3).map(d=>`<div class="dc"><div class="dp">${d.overlap_percentage}%</div>
    <div><div style="font-weight:700;color:var(--navy)">${e(d.tool_a)} &#8596; ${e(d.tool_b)}</div>
    <div style="font-size:12px;color:var(--mut);line-height:1.5">${e(d.rationale)}</div>
    <div style="font-size:12px;color:var(--green);font-weight:700;margin-top:3px">Est. savings: $${fmt(d.potential_annual_savings||0)}/yr &bull; ${d.priority} Priority</div></div></div>`).join('')
    :'<p class="mu2 tx">No duplications detected.</p>';
  rPrev();
}

function rDonut(cnt){
  const ctx=document.getElementById('cd');if(!ctx)return;
  if(cD)cD.destroy();
  const CLR={Retain:'#00A651',Rehost:'#0063DC',Replatform:'#FFC200',Refactor:'#FF7700',Replace:'#E31837',Retire:'#6B7A99',TBD:'#aaa'};
  const lbs=Object.keys(cnt),data=Object.values(cnt);
  cD=new Chart(ctx,{type:'doughnut',data:{labels:lbs,datasets:[{data,backgroundColor:lbs.map(l=>CLR[l]||'#ccc'),borderWidth:2,borderColor:'#fff'}]},
    options:{cutout:'66%',plugins:{legend:{position:'right',labels:{boxWidth:11,font:{size:11}}}},animation:{duration:500}}});
}
function rBar(){
  const ctx=document.getElementById('cb');if(!ctx)return;
  if(cB)cB.destroy();
  const s=[...TOOLS].sort((a,b)=>(b.composite_score||0)-(a.composite_score||0)).slice(0,10);
  cB=new Chart(ctx,{type:'bar',data:{
    labels:s.map(t=>t.name.length>13?t.name.slice(0,12)+'…':t.name),
    datasets:[{label:'Score',data:s.map(t=>t.composite_score||0),
      backgroundColor:s.map(t=>(t.composite_score||0)>=7?'#00A651':(t.composite_score||0)>=5?'#FFC200':'#E31837'),borderRadius:4}]},
    options:{indexAxis:'y',plugins:{legend:{display:false}},
      scales:{x:{min:0,max:10,grid:{color:'#f0f0f0'}},y:{grid:{display:false},ticks:{font:{size:11}}}},animation:{duration:500}}});
}

function renderInv(flt){
  const tools=flt||TOOLS;
  document.getElementById('ib').innerHTML=tools.map(t=>`<tr>
    <td><strong>${e(t.name)}</strong></td><td>${e(t.vendor||'—')}</td><td>${e(t.category)}</td>
    <td>${e(t.deployment||'—')}</td><td>${t.annual_cost?'$'+fmt(t.annual_cost):'—'}</td>
    <td>${t.user_count??'—'}</td>
    <td><div class="sb2"><div class="bt2"><div class="bf2" style="width:${(t.composite_score||0)*10}%"></div></div>
    <span class="bl2 ${sc(t.composite_score)}">${t.composite_score||'—'}</span></div></td>
    <td>${((t.scores||{}).risk_score||'—').toFixed?((t.scores||{}).risk_score||0).toFixed(1):'—'}</td>
    <td>${bdg(t.confidence_level||'—')}</td><td>${bdg(t.rationalization_action||'TBD')}</td></tr>`).join('');
  if(!flt){
    const cats=[...new Set(TOOLS.map(t=>t.category))].sort();
    const sel=document.getElementById('ic3');
    sel.innerHTML='<option value="">All Categories</option>'+cats.map(c=>`<option>${c}</option>`).join('');
    setupFlt();
  }
  document.getElementById('db2').innerHTML=DUPS.length
    ?DUPS.map(d=>`<tr><td>${e(d.category)}</td><td><strong>${e(d.tool_a)}</strong></td><td><strong>${e(d.tool_b)}</strong></td>
    <td><strong>${d.overlap_percentage}%</strong></td><td>${e(d.retain_candidate)}</td>
    <td>$${fmt(d.potential_annual_savings||0)}</td><td>${bdg(d.priority)}</td></tr>`).join('')
    :'<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--mut)">No duplications detected.</td></tr>';
}
function setupFlt(){
  const s=document.getElementById('is'),c=document.getElementById('ic3'),a=document.getElementById('ia');
  const apply=()=>{const q=s.value.toLowerCase(),cat=c.value,act=a.value;
    renderInv(TOOLS.filter(t=>(!q||t.name.toLowerCase().includes(q)||(t.vendor||'').toLowerCase().includes(q))&&(!cat||t.category===cat)&&(!act||t.rationalization_action===act)));};
  s.oninput=apply;c.onchange=apply;a.onchange=apply;
}

async function runAssess(){
  if(!TOOLS.length)return toast('Ingest your tool inventory first','er');
  showLd('AI Agent is analysing your portfolio...');
  try{
    const r=await fetch(A+'/api/assess',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tools:TOOLS,duplications:DUPS,
        industry:document.getElementById('ai2').value,
        focus:document.getElementById('af').value||null})});
    const d=await r.json(); hideLd();
    if(!r.ok)return toast(d.detail||'Assessment failed','er');
    ASSESS=d.assessment; rAssess(ASSESS); rPrev(); toast('Assessment complete','ok');
  }catch(ex){hideLd();toast('Error: '+ex.message,'er');}
}

function rAssess(a){
  const el=document.getElementById('ar'); let html='';
  if(a.executive_summary)html+=`<div class="card m4"><div class="ch"><span>&#128203;</span><span class="ct">Executive Summary</span></div><div class="eb">${e(a.executive_summary)}</div></div>`;
  const po=a.portfolio_overview;
  if(po){const hc=po.portfolio_health==='Healthy'?'green':po.portfolio_health==='At Risk'?'amber':'red';
    html+=`<div class="card m4"><div class="ch"><span>&#128202;</span><span class="ct">Portfolio Overview</span></div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
    <div><div style="font-size:22px;font-weight:800;color:var(--blue)">${po.total_tools||0}</div><div class="tx mu2">Total Tools</div></div>
    <div><div style="font-size:22px;font-weight:800;color:var(--blue)">$${fmt(po.total_annual_cost||0)}</div><div class="tx mu2">Annual Spend</div></div>
    <div><div style="font-size:22px;font-weight:800;color:var(--${hc})">${po.portfolio_health||'—'}</div><div class="tx mu2">Portfolio Health</div></div>
    </div>${po.health_rationale?`<p class="tx mu2 mt4">${e(po.health_rationale)}</p>`:''}</div>`;}
  const recs=a.top_recommendations||[];
  if(recs.length)html+=`<div class="card m4"><div class="ch"><span>&#127919;</span><span class="ct">Top Priority Recommendations</span></div>
    ${recs.slice(0,5).map(r=>`<div class="rc"><div class="rh">
    <div class="rr">${r.rank||'#'}</div><div style="font-weight:700;color:var(--navy);flex:1">${e(r.title||'')}</div>
    ${bdg(r.priority||'Medium')} ${bdg(r.confidence||'Medium')}</div>
    <p style="font-size:13px;color:#444;line-height:1.6;margin-bottom:5px">${e(r.description||'')}</p>
    <p class="tx mu2">Impact: ${e(r.impact||'')} &bull; ${r.timeline||''}</p></div>`).join('')}</div>`;
  const rm=a.roadmap||{};
  if(rm.short_term||rm.medium_term||rm.long_term){
    const ph=(key)=>{const items=Array.isArray(rm[key])?rm[key]:[rm[key]].filter(Boolean);
      return items.map(i=>`<div class="pi"><div class="pd"></div><div>${e(String(i))}</div></div>`).join('');};
    html+=`<div class="card m4"><div class="ch"><span>&#128508;&#65039;</span><span class="ct">Rationalization Roadmap</span></div>
    <div class="rmc">
    <div class="ph p1"><div class="pht">Phase 1 &mdash; Quick Wins (0&ndash;3 Months)</div>${ph('short_term')}</div>
    <div class="ph p2"><div class="pht">Phase 2 &mdash; Strategic (3&ndash;12 Months)</div>${ph('medium_term')}</div>
    <div class="ph p3"><div class="pht">Phase 3 &mdash; Transformation (12&ndash;24 Months)</div>${ph('long_term')}</div>
    </div></div>`;}
  const risks=a.risk_highlights||[];
  if(risks.length)html+=`<div class="card m4"><div class="ch"><span>&#9888;&#65039;</span><span class="ct">Risk Highlights</span></div>
    ${risks.map(r=>`<div class="dc" style="background:#fff5f5;border-color:#fad7d7">
    <div style="font-size:18px">${r.severity==='Critical'?'🔴':r.severity==='High'?'🟠':'🟡'}</div>
    <div><div style="font-weight:700;color:var(--navy)">${r.risk_type} &mdash; ${r.severity}</div>
    <div style="font-size:13px;color:#666;margin:3px 0">${e(r.description||'')}</div>
    <div style="font-size:12px;color:var(--green)">Mitigation: ${e(r.mitigation||'')}</div></div></div>`).join('')}</div>`;
  const oc=a.expected_outcomes||{};
  if(Object.keys(oc).length)html+=`<div class="card m4"><div class="ch"><span>&#9989;</span><span class="ct">Expected Business Outcomes</span></div>
    <div class="tw"><table><tbody>${Object.entries(oc).map(([k,v])=>`<tr><td><strong>${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</strong></td><td>${e(String(v))}</td></tr>`).join('')}</tbody></table></div></div>`;
  el.innerHTML=html||'<p class="mu2">No assessment data yet.</p>';
}

function setupChat(){
  const inp=document.getElementById('cin'),btn=document.getElementById('bsnd');
  btn.onclick=sendChat;
  inp.onkeydown=ev=>{if(ev.key==='Enter'&&!ev.shiftKey){ev.preventDefault();sendChat();}};
  document.getElementById('chs').onclick=ev=>{if(ev.target.classList.contains('chip')){inp.value=ev.target.textContent;sendChat();}};
}
async function sendChat(){
  const inp=document.getElementById('cin'); const msg=inp.value.trim(); if(!msg)return;
  addMsg('user',msg); inp.value=''; HIST.push({role:'user',content:msg});
  const typ=addMsg('assistant','…',true);
  try{
    const r=await fetch(A+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,history:HIST.slice(-20),tools:TOOLS})});
    const d=await r.json(); typ.remove();
    const rep=d.reply||'Sorry, I could not generate a response.';
    addMsg('assistant',rep); HIST.push({role:'assistant',content:rep});
  }catch(ex){typ.remove();addMsg('assistant','Error: '+ex.message);}
}
function addMsg(role,content,isTyping=false){
  const c=document.getElementById('cm'); const d=document.createElement('div');
  d.className='msg '+(role==='user'?'mu':'ma');
  const bub=isTyping?'<span style="opacity:.5">&#8230;</span>':mdt(content);
  d.innerHTML=`<div class="mav">${role==='user'?'U':'AI'}</div><div class="mb2">${bub}</div>`;
  c.appendChild(d); c.scrollTop=c.scrollHeight; return d;
}
function mdt(t){
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>').replace(/^## (.+)$/gm,'<h3>$1</h3>')
    .replace(/^- (.+)$/gm,'<li>$1</li>').replace(/\n\n/g,'<br><br>').replace(/\n/g,'<br>');
}

async function dlReport(){
  if(!TOOLS.length)return toast('Ingest tools first','er');
  showLd('Generating report...');
  try{
    const r=await fetch(A+'/api/report',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tools:TOOLS,duplications:DUPS,assessment:ASSESS})});
    const d=await r.json(); hideLd();
    if(!r.ok)return toast('Report generation failed','er');
    const blob=new Blob([d.html],{type:'text/html'});
    const url=URL.createObjectURL(blob); const a=document.createElement('a');
    a.href=url; a.download='tech_rationalization_'+new Date().toISOString().slice(0,10)+'.html';
    a.click(); URL.revokeObjectURL(url); toast('Report downloaded','ok');
  }catch(ex){hideLd();toast('Error: '+ex.message,'er');}
}

function rPrev(){
  const el=document.getElementById('rp'); if(!TOOLS.length)return;
  const tc=TOOLS.reduce((s,t)=>s+(t.annual_cost||0),0);
  const cnt={};TOOLS.forEach(t=>{const a=t.rationalization_action||'TBD';cnt[a]=(cnt[a]||0)+1;});
  el.innerHTML=`<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    <div style="background:var(--bg);padding:14px;border-radius:8px;text-align:center"><div style="font-size:22px;font-weight:800;color:var(--blue)">${TOOLS.length}</div><div class="tx mu2">Tools</div></div>
    <div style="background:var(--bg);padding:14px;border-radius:8px;text-align:center"><div style="font-size:22px;font-weight:800;color:var(--blue)">$${fmt(tc)}</div><div class="tx mu2">Annual Spend</div></div></div>
    <div style="margin-top:12px">${Object.entries(cnt).map(([a,c])=>`<div class="fx jb ic2" style="padding:7px 0;border-bottom:1px solid var(--bdr)">${bdg(a)}<strong>${c} tools</strong></div>`).join('')}</div>
    ${ASSESS.executive_summary?`<div style="margin-top:12px;background:var(--bg);padding:12px;border-radius:7px;font-size:12px;line-height:1.6;color:var(--mut)">${e(ASSESS.executive_summary.slice(0,280))}&#8230;</div>`:''}`;
}

// Helpers
function bdg(a){const MAP={Retain:'br',Rehost:'bh',Replatform:'bp',Refactor:'bf',Replace:'bl',Retire:'bt',High:'bhi',Medium:'bme',Low:'blo',Critical:'bcr'};return `<span class="b ${MAP[a]||'bme'}">${a}</span>`;}
function fmt(n){return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':Math.round(n).toLocaleString();}
function e(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function sc(s){return !s&&s!==0?'':s>=7?'sh':s>=5?'sm2':'sl2';}
function showLd(m='Processing...'){document.getElementById('lm').textContent=m;document.getElementById('ld').classList.add('show');}
function hideLd(){document.getElementById('ld').classList.remove('show');}
function toast(msg,type='in'){
  const c=document.getElementById('ts2'); const t=document.createElement('div');
  t.className='tt2 '+type;
  t.innerHTML=`<span>${{ok:'&#10003;',er:'&#10007;',in:'&#8505;'}[type]||'&#8505;'}</span><span>${e(msg)}</span>`;
  c.appendChild(t); setTimeout(()=>t.remove(),4000);
}
document.addEventListener('DOMContentLoaded',init);
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set!")
        print("   Create a .env file with:  ANTHROPIC_API_KEY=sk-ant-...\n")
    else:
        print(f"\n✅ Anthropic API key loaded | Model: {ANTHROPIC_MODEL}")

    print("🚀 Starting Tech Rationalization AI Agent...")
    print("📍 Open your browser at: http://localhost:8000\n")

    # Auto-open browser after 1.5 seconds
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="warning")
