"""
Tech Rationalization AI Agent -- Vercel-ready, stateless FastAPI backend.
All state is managed by the client (browser). No server-side sessions.

Local dev:  uvicorn api.index:app --reload --port 8000
Vercel:     vercel deploy
"""

import json, uuid, io, os
from typing import Optional, List, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(title="PlatformAssessor AI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an Enterprise Technology Strategy AI Agent specializing in Platform, Application, and Tools Assessment & Rationalization, aligned with KPMG-style consulting frameworks and telecom/large enterprise transformation programs (e.g., Airtel, Tanla-like environments).

You combine the expertise of:
- Senior Enterprise Architect
- Technology Due-Diligence Consultant
- CIO/CTO Advisory Analyst
- Telecom Digital Transformation Expert

PRIMARY OBJECTIVE: Maximize business value, reduce cost, mitigate risk, and simplify the technology landscape by assessing tools across 7 dimensions and applying the 6R Rationalization model.

6R MODEL:
- Retain    -> Strategic, healthy, high-value (Score >=7.5, Risk <=4)
- Rehost    -> Lift-and-shift to cloud (Score >=6, on-prem, cloud-ready)
- Replatform -> Minor modernization (Score 5-7)
- Refactor  -> Significant redesign needed (Score 3-5)
- Replace   -> Better alternative exists (Score <5 or cost-inefficient)
- Retire    -> Decommission -- redundant, EOL, or very low value

SCORING DIMENSIONS (0-10 each):
1. Business Value -- Strategic importance, revenue impact, criticality
2. Adoption Rate -- User adoption %, utilization signals
3. Integration Depth -- API dependencies, systemic coupling
4. Vendor Support -- Roadmap clarity, vendor health, EOL status
5. Cost Efficiency -- Cost-per-user vs market benchmarks
6. Technical Health -- Modernity, tech debt, maintenance burden
7. Risk Score -- Security, compliance, obsolescence, vendor lock-in (higher = more risky)

TELECOM FILTERS (apply when telecom/TMT context): Latency sensitivity, 24x7 SLA, TRAI/GDPR compliance, transaction volume at scale, customer SLA impact.

BEHAVIOR: Be data-driven. State assumptions explicitly. Tag recommendations with confidence: High/Medium/Low. Provide trade-offs, not just conclusions. Use executive-grade language. Never give generic recommendations.

OUTPUT: Always produce structured recommendations with rationale, impact analysis, roadmap (0-3 months | 3-12 months | 12-24 months), and expected outcomes."""


# ═══════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# DUPLICATION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA INGESTION (in-memory)
# ═══════════════════════════════════════════════════════════════════════════
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
    # additional name aliases
    "application_name":"name","product_name":"name","product":"name",
    "system_name":"name","system":"name","software":"name","solution":"name",
    "service_name":"name","service":"name","asset_name":"name","asset":"name",
    "technology":"name","tech_name":"name","component":"name",
    # additional cost aliases
    "total_cost":"annual_cost","yearly_cost":"annual_cost","license_cost":"annual_cost",
    "license_fee":"annual_cost","subscription_cost":"annual_cost","subscription":"annual_cost",
    "spend":"annual_cost","budget":"annual_cost","annual_budget":"annual_cost",
    "contract_value":"annual_cost","contract_cost":"annual_cost",
    # additional user aliases
    "seats":"user_count","licenses":"user_count","license_count2":"user_count","headcount":"user_count",
    # additional category aliases
    "tool_type":"category","app_type":"category","type2":"category",
    # additional vendor aliases
    "manufacturer":"vendor","provider":"vendor","publisher":"vendor","oem":"vendor",
    # additional age aliases
    "years_in_use":"age_years","tool_age":"age_years","implementation_year":"age_years",
    # additional integration aliases
    "number_of_integrations":"integrations","num_integrations":"integrations","api_count":"integrations",
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
    # Fallback: detect name column if not found
    if "name" not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype == object and c not in
                    ("vendor","category","description","owner","business_unit","deployment",
                     "criticality","license_type","end_of_life","compliance_required")]
        if str_cols:
            best = max(str_cols, key=lambda c: df[c].nunique())
            df = df.rename(columns={best: "name"})
    df=df.where(pd.notna(df),None)
    return [normalize(r.to_dict()) for _,r in df.iterrows()]


# ═══════════════════════════════════════════════════════════════════════════
# AI HELPERS
# ═══════════════════════════════════════════════════════════════════════════
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
            f"""Extract all tools/apps/platforms from the text. Return ONLY valid JSON with a "tools" array -- no other text or markdown.
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

    tool_rows="".join(f"<tr><td><b>{t.get('name','')}</b></td><td>{t.get('vendor') or '--'}</td>"
        f"<td>{t.get('category','')}</td><td>{'${:,.0f}'.format(t.get('annual_cost',0)) if t.get('annual_cost') else '--'}</td>"
        f"<td>{t.get('user_count') or '--'}</td><td><b>{t.get('composite_score','--')}</b>/10</td>"
        f"<td>{t.get('scores',{}).get('risk_score','--')}/10</td><td>{bdg(t.get('rationalization_action','TBD'))}</td></tr>" for t in tools)

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
<div style="font-size:18px">{'[CRITICAL]' if r.get('severity')=='Critical' else '[HIGH]' if r.get('severity')=='High' else '[MED]'}</div>
<div><div style="font-weight:700;color:#1a2340">{r.get('risk_type')} -- {r.get('severity')}</div>
<div style="font-size:13px;color:#666;margin:3px 0">{r.get('description','')}</div>
<div style="font-size:12px;color:#00A651">Mitigation: {r.get('mitigation','')}</div></div></div>""" for r in risks)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Tech Rationalization Report -- {datetime.now().strftime('%B %Y')}</title>
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
<p>Enterprise Platform, Application &amp; Tools Assessment · AI-Powered Advisory · 6R Rationalization Framework</p>
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
<div class="ph" style="border-top:3px solid #00A651"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#00A651;margin-bottom:10px">Phase 1 -- Quick Wins (0-3 Months)</p><ul>{ph_items('short_term')}</ul></div>
<div class="ph" style="border-top:3px solid #0063DC"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#0063DC;margin-bottom:10px">Phase 2 -- Strategic (3-12 Months)</p><ul>{ph_items('medium_term')}</ul></div>
<div class="ph" style="border-top:3px solid #7B2FBE"><p style="font-size:11px;font-weight:700;text-transform:uppercase;color:#7B2FBE;margin-bottom:10px">Phase 3 -- Transformation (12-24 Months)</p><ul>{ph_items('long_term')}</ul></div>
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


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
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
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════
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
async def health():
    return {"status":"healthy","model":ANTHROPIC_MODEL,"key_set":bool(ANTHROPIC_API_KEY)}
from mangum import Mangum
handler = Mangum(app)
