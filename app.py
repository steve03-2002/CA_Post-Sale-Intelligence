import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Post-Sale Revenue Intelligence", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.stApp{background:#0e1117;color:#f5f7fb}
section[data-testid="stSidebar"]{background:#252831}
.block-container{padding-top:1.6rem;max-width:1600px}
.info-banner{background:#172f48;color:#a8d4ff;padding:13px 16px;border-radius:8px;margin:12px 0 18px}
.metric-note{color:#a5abb5;font-size:.83rem}
div[data-testid="stDataFrame"]{border:1px solid #30343b;border-radius:8px}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "delivery_to_revenue_synthetic_data.xlsx"

# Synthetic dataset generator used only if the bundled workbook is missing.
def make_sample(n=1000, seed=20261027):
    rng=np.random.default_rng(seed)
    industries=["Manufacturing","IT Services","Logistics","Healthcare","Retail","BFSI","Education","Pharma","FMCG","Engineering"]
    owners=["Priya","Arjun","Divya","Karthik","Meera","Rahul","Ananya","Vikram"]
    prefixes=["Kaveri","Vaigai","Meenakshi","Chola","Pandian","Nilgiri","Sangam","Thirumala","Marina","Cauvery"]
    suffixes=["Industries","Enterprises","Systems","Group","Works","Services","Solutions","Technologies"]
    rows=[]
    for i in range(1,n+1):
        grp=rng.choice(["healthy","watch","risk"],p=[.55,.25,.20])
        if grp=="healthy":
            prev=int(rng.integers(1,9)); tickets=max(0,prev+int(rng.integers(-2,4)))
            breaches=int(rng.choice([0,1,2],p=[.8,.18,.02])); esc=int(rng.choice([0,1],p=[.9,.1]))
            csat=round(float(rng.uniform(3.8,5)),1); slipped=int(rng.choice([0,1],p=[.85,.15]))
            adoption=int(rng.integers(65,101)); renewal=int(rng.integers(15,366)); contact=int(rng.integers(1,101))
        elif grp=="watch":
            prev=int(rng.integers(3,9)); tickets=prev+int(rng.integers(0,6))
            breaches=int(rng.integers(0,3)); esc=int(rng.integers(0,2)); csat=round(float(rng.uniform(3,4.2)),1)
            slipped=int(rng.integers(0,3)); adoption=int(rng.integers(40,76)); renewal=int(rng.integers(15,366)); contact=int(rng.integers(15,101))
        else:
            prev=int(rng.integers(4,13)); tickets=prev+int(rng.integers(2,13))
            breaches=int(rng.integers(1,5)); esc=int(rng.integers(0,4)); csat=round(float(rng.uniform(1.5,3.5)),1)
            slipped=int(rng.integers(1,5)); adoption=int(rng.integers(15,61)); renewal=int(rng.integers(5,151)); contact=int(rng.integers(35,101))
        modules=rng.choice(["Core HR","Attendance","Payroll","Recruitment","Performance"],size=int(rng.integers(2,5)),replace=False)
        rows.append({
            "account_id":f"ACC-{i:04d}","account_name":f"{rng.choice(prefixes)} {rng.choice(suffixes)} {i:04d}",
            "industry":rng.choice(industries),"contract_value":int(rng.integers(50,4001))*1000,
            "tenure_months":int(rng.integers(3,73)),"renewal_in_days":renewal,
            "modules_owned":", ".join(modules),"last_contact_days":contact,"account_owner":rng.choice(owners),
            "tickets_30d":tickets,"tickets_prev_30d":prev,"sla_breaches_30d":breaches,"escalations_30d":esc,
            "csat":csat,"milestones_slipped":slipped,"open_change_requests":int(rng.integers(0,7)),
            "adoption_pct":adoption
        })
    return pd.DataFrame(rows)

def score_accounts(raw, high_threshold, watch_threshold, renewal_window, margin, horizon_years, retention_success, expansion_win):
    df=raw.copy()
    numeric=["contract_value","tenure_months","renewal_in_days","last_contact_days","tickets_30d","tickets_prev_30d",
             "sla_breaches_30d","escalations_30d","csat","milestones_slipped","open_change_requests","adoption_pct"]
    for c in numeric:
        if c not in df: df[c]=0
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    if "account_id" not in df: df["account_id"]=[f"ROW-{i+1:04d}" for i in range(len(df))]
    if "account_name" not in df: df["account_name"]=df["account_id"].astype(str)
    if "industry" not in df: df["industry"]="Unknown"
    if "account_owner" not in df: df["account_owner"]="Unassigned"
    if "modules_owned" not in df: df["modules_owned"]=""
    clamp=lambda x,lo=0,hi=1: np.minimum(hi,np.maximum(lo,x))
    signals={
        "Rising ticket volume":clamp((df.tickets_30d-df.tickets_prev_30d)/np.maximum(df.tickets_prev_30d,1),0,1),
        "SLA breaches":clamp(df.sla_breaches_30d/3),
        "Escalations":clamp(df.escalations_30d/3),
        "Low CSAT":clamp((4.2-df.csat)/2.7),
        "Slipped milestones":clamp(df.milestones_slipped/3),
        "Low adoption":clamp((60-df.adoption_pct)/45),
        "No recent contact":clamp(df.last_contact_days/90),
        "Renewal approaching":clamp(1-df.renewal_in_days/120)
    }
    weights={"Rising ticket volume":1.0,"SLA breaches":1.4,"Escalations":1.1,"Low CSAT":1.2,
             "Slipped milestones":1.1,"Low adoption":1.4,"No recent contact":.7,"Renewal approaching":.7}
    total=sum(signals[k]*v for k,v in weights.items())
    denom=sum(weights.values())
    df["risk_score"]=(100*total/denom).round(1)
    df["risk_score_pct"]=df["risk_score"]
    df["risk_drivers"]=df.apply(lambda r: ", ".join([
        name for name, condition in [
            ("Rising tickets",r.tickets_30d>r.tickets_prev_30d),("SLA breaches",r.sla_breaches_30d>0),
            ("Escalations",r.escalations_30d>0),("Low CSAT",r.csat<3.5),("Low adoption",r.adoption_pct<50),
            ("Slipped milestones",r.milestones_slipped>0),("Renewal soon",r.renewal_in_days<=renewal_window),
            ("No recent contact",r.last_contact_days>=60)] if condition
    ]) or "No major flagged signal",axis=1)
    # Expected value model: clearly labelled scenario estimate, not calibrated probability.
    # Remaining lifetime value proxy uses annual contract value * assumed remaining years.
    df["remaining_value_proxy"]=(df.contract_value*np.maximum(horizon_years-(df.tenure_months/12),0.5)).round(0)
    df["risk_index_fraction"]=df.risk_score/100
    df["estimated_at_risk_value"]=(df.risk_index_fraction*df.remaining_value_proxy).round(0)
    df["expected_retention_value"]=(df.estimated_at_risk_value*retention_success).round(0)
    df["intervention_cost"]=np.select(
        [df.risk_score>=high_threshold,df.risk_score>=watch_threshold],
        [np.where(df.sla_breaches_30d>=2,45000,30000),12000],default=0
    )
    df["expected_net_retention_value"]=(df.expected_retention_value-df.intervention_cost).round(0)
    df["expansion_score"]=(
        .35*clamp((df.adoption_pct-60)/35)+.25*(1-clamp(df.tickets_30d/10))+
        .20*df.milestones_slipped.eq(0).astype(float)+.20*df.csat.ge(4.3).astype(float)
    ).round(2)
    def segment(r):
        if r.risk_score>=high_threshold: return "HIGH RISK"
        if r.risk_score>=watch_threshold: return "WATCH"
        if r.expansion_score>=.65 and r.risk_score<watch_threshold: return "EXPANSION"
        return "HEALTHY"
    df["segment"]=df.apply(segment,axis=1)
    def action(r):
        if r.segment=="EXPANSION": return "Assess add-on module"
        if r.sla_breaches_30d>=2 or r.escalations_30d>=2: return "SLA remediation plan"
        if r.adoption_pct<50 or r.milestones_slipped>=2: return "Adoption workshop"
        if r.last_contact_days>=60 or r.csat<3: return "Executive sponsor call"
        if r.segment in ["HIGH RISK","WATCH"]: return "Customer check-in"
        return "Monitor"
    df["recommended_action"]=df.apply(action,axis=1)
    df["expansion_value_proxy"]=(df.contract_value*df.expansion_score*expansion_win*.25).round(0)
    df["expected_net_value"]=(df.expected_net_retention_value+df.expansion_value_proxy).round(0)
    for name,w in weights.items():
        key="contrib_"+name.lower().replace(" ","_")
        df[key]=(100*signals[name]*w/denom).round(1)
    df["approval_status"]=np.where(df.risk_score>=high_threshold,"Human approval required",
                          np.where(df.risk_score>=watch_threshold,"Review recommended","Monitor"))
    return df

st.sidebar.title("Data input")
uploaded=st.sidebar.file_uploader("Upload account dataset (.csv or .xlsx)",type=["csv","xlsx"])
use_sample=st.sidebar.checkbox("Use included synthetic sample when no file is uploaded",value=True)
st.sidebar.caption("Synthetic records are for prototype testing, not evidence of real churn behaviour.")
st.sidebar.divider()
st.sidebar.subheader("Decision assumptions")
high_threshold=st.sidebar.slider("High-risk score threshold",40,85,65)
watch_threshold=st.sidebar.slider("Watch threshold",20,60,40)
renewal_window=st.sidebar.slider("Renewal warning window (days)",30,120,60)
margin=st.sidebar.slider("Assumed contribution margin (%)",10,60,30)/100
horizon_years=st.sidebar.slider("Remaining-value horizon (years)",1,5,3)
retention_success=st.sidebar.slider("Assumed intervention success rate (%)",10,80,35)/100
expansion_win=st.sidebar.slider("Assumed expansion win rate (%)",5,60,20)/100
st.sidebar.caption("All assumptions are editable scenarios, not validated rates.")

if uploaded:
    try:
        raw=pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        source=f"Uploaded file: {uploaded.name}"
    except Exception as e:
        st.error(f"Unable to read file: {e}"); st.stop()
elif use_sample:
    try: raw=pd.read_excel(DATA_PATH,sheet_name="Combined_Data") if DATA_PATH.exists() else make_sample()
    except Exception: raw=make_sample()
    source="Included synthetic sample"
else:
    st.info("Upload a CSV/XLSX file or enable the synthetic sample."); st.stop()
if raw.empty:
    st.error("The dataset has no rows."); st.stop()
df=score_accounts(raw,high_threshold,watch_threshold,renewal_window,margin,horizon_years,retention_success,expansion_win)

st.title("🧭 POST-SALE REVENUE INTELLIGENCE")
st.header("Cross-System Risk, Retention & Expansion Prioritisation")
st.caption("Customer Analytics Final Project | Transparent rules | Human approval before CRM action")
st.markdown('<div class="info-banner">The gap addressed: connect post-sale signals, rank accounts by commercial exposure, recommend a response and route high-impact actions for approval. This prototype uses a weighted risk index, not a calibrated churn probability, and does not modify a live CRM.</div>',unsafe_allow_html=True)

flagged=df[df.segment.isin(["HIGH RISK","WATCH","EXPANSION"])]
retention=df[df.segment.isin(["HIGH RISK","WATCH"])]
growth=df[df.segment=="EXPANSION"]
top=df.sort_values("expected_net_value",ascending=False).iloc[0]
total_contract=df.contract_value.sum()
total_exposure=df.estimated_at_risk_value.sum()
expected_retention=df.expected_retention_value.sum()
expansion_pipe=df.expansion_value_proxy.sum()
intervention_spend=df.intervention_cost.sum()
net_value=df.expected_net_value.sum()
k=st.columns(5)
k[0].metric("Accounts analysed",f"{len(df):,}")
k[1].metric("Accounts needing action",f"{len(flagged):,}")
k[2].metric("Estimated value at risk",f"₹{total_exposure:,.0f}")
k[3].metric("Expected net value (scenario)",f"₹{net_value:,.0f}")
k[4].metric("Expansion candidates",f"{len(growth):,}")
st.caption(f"Data source: {source}. Scenario assumptions: {retention_success:.0%} intervention success, {expansion_win:.0%} expansion win rate, {margin:.0%} contribution margin, {horizon_years}-year horizon. These are illustrative assumptions.")

tabs=st.tabs(["Overview","Value-weighted account analysis","Cross-system signals","Decision & approval queue","Expansion opportunities","Data quality & assumptions"])
with tabs[0]:
    left,right=st.columns([1.2,1])
    with left:
        st.subheader("Priority ranking by expected net value")
        view=df.sort_values("expected_net_value",ascending=False).head(15).copy()
        for c in ["contract_value","remaining_value_proxy","estimated_at_risk_value","expected_retention_value","intervention_cost","expected_net_value"]:
            view[c]=view[c].map(lambda x:f"₹{x:,.0f}")
        st.dataframe(view[["account_name","segment","risk_score","contract_value","estimated_at_risk_value","expected_net_value","recommended_action"]].rename(columns={
            "account_name":"Account","segment":"Segment","risk_score":"Risk index /100","contract_value":"Annual contract value",
            "estimated_at_risk_value":"Estimated value at risk","expected_net_value":"Expected net value (scenario)","recommended_action":"Recommended action"
        }),use_container_width=True,hide_index=True)
    with right:
        st.subheader("Commercial exposure by segment")
        agg=df.groupby("segment",as_index=False).agg(accounts=("account_id","count"),exposure=("estimated_at_risk_value","sum"))
        chart=alt.Chart(agg).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("segment:N",title=None,sort="-y"),y=alt.Y("exposure:Q",title="Estimated value at risk (₹)"),
            tooltip=["segment","accounts",alt.Tooltip("exposure:Q",format=",.0f")]
        ).properties(height=330)
        st.altair_chart(chart,use_container_width=True)
    a,b,c,d=st.columns(4)
    a.metric("Annual contract value",f"₹{total_contract:,.0f}")
    b.metric("Expected retention value",f"₹{expected_retention:,.0f}")
    c.metric("Intervention cost scenario",f"₹{intervention_spend:,.0f}")
    d.metric("Expansion value pipeline",f"₹{expansion_pipe:,.0f}")

with tabs[1]:
    st.subheader("Prioritise accounts by risk and commercial value")
    c1,c2,c3=st.columns(3)
    segs=c1.multiselect("Segments",sorted(df.segment.unique()),default=sorted(df.segment.unique()))
    minvalue=c2.number_input("Minimum estimated value at risk (₹)",min_value=0,value=0,step=100000)
    minscore=c3.slider("Minimum risk index",0,100,0)
    f=df[df.segment.isin(segs)&(df.estimated_at_risk_value>=minvalue)&(df.risk_score>=minscore)].sort_values("expected_net_value",ascending=False)
    st.dataframe(f[["account_id","account_name","industry","account_owner","segment","risk_score","contract_value","remaining_value_proxy","estimated_at_risk_value","expected_retention_value","intervention_cost","expected_net_value","risk_drivers","recommended_action"]],use_container_width=True,hide_index=True)
    st.download_button("Download prioritised accounts CSV",f.to_csv(index=False).encode(), "prioritised_accounts.csv","text/csv")

with tabs[2]:
    st.subheader("Cross-system warning signals")
    sig=pd.DataFrame([
        ["Rising ticket volume",int((df.tickets_30d>df.tickets_prev_30d).sum())],
        ["SLA breaches",int((df.sla_breaches_30d>0).sum())],
        ["Escalations",int((df.escalations_30d>0).sum())],
        ["Low CSAT (<3.5)",int((df.csat<3.5).sum())],
        ["Low adoption (<50%)",int((df.adoption_pct<50).sum())],
        ["Slipped milestones",int((df.milestones_slipped>0).sum())],
        [f"Renewal within {renewal_window} days",int((df.renewal_in_days<=renewal_window).sum())],
        ["No contact for 60+ days",int((df.last_contact_days>=60).sum())]
    ],columns=["Signal","Accounts"])
    ch=alt.Chart(sig).mark_bar().encode(x=alt.X("Accounts:Q"),y=alt.Y("Signal:N",sort="-x"),tooltip=["Signal","Accounts"]).properties(height=350)
    st.altair_chart(ch,use_container_width=True)
    st.subheader("Risk index versus adoption")
    sc=alt.Chart(df).mark_circle(size=55,opacity=.65).encode(
        x=alt.X("adoption_pct:Q",title="Platform adoption (%)"),y=alt.Y("risk_score:Q",title="Weighted risk index (0–100)"),
        color=alt.Color("segment:N"),tooltip=["account_name","segment","adoption_pct","risk_score","estimated_at_risk_value","risk_drivers"]
    ).interactive().properties(height=360)
    st.altair_chart(sc,use_container_width=True)
    st.caption("Counts overlap: an account can exhibit several warning signals.")

with tabs[3]:
    st.subheader("Recommended actions and approval routing")
    queue=df[df.segment.isin(["HIGH RISK","WATCH","EXPANSION"])].sort_values(["approval_status","expected_net_value"],ascending=[True,False]).copy()
    st.dataframe(queue[["account_id","account_name","account_owner","segment","risk_score","estimated_at_risk_value","expected_net_value","recommended_action","approval_status","risk_drivers"]],use_container_width=True,hide_index=True)
    with st.expander("Explain a selected account's risk score"):
        chosen=st.selectbox("Account",df.account_name.tolist(),key="explain_account")
        row=df.loc[df.account_name==chosen].iloc[0]
        contrib=[c for c in df.columns if c.startswith("contrib_")]
        cd=pd.DataFrame([{"Signal":c.replace("contrib_","").replace("_"," ").title(),"Score contribution (points)":row[c]} for c in contrib]).sort_values("Score contribution (points)",ascending=False)
        st.write(f"**{row.account_name}** — weighted risk index **{row.risk_score:.1f}/100**. This is not a calibrated probability.")
        st.dataframe(cd,use_container_width=True,hide_index=True)
    st.info("This queue simulates recommendations and approval routing. It does not create Zoho tasks or send outreach.")
    st.download_button("Download approval queue CSV",queue.to_csv(index=False).encode(),"approval_queue.csv","text/csv")

with tabs[4]:
    st.subheader("Expansion candidates")
    exp=df[df.segment=="EXPANSION"].sort_values("expansion_value_proxy",ascending=False).copy()
    st.write(f"Candidates: **{len(exp):,}** | Scenario expansion pipeline: **₹{exp.expansion_value_proxy.sum():,.0f}**")
    st.dataframe(exp[["account_id","account_name","industry","account_owner","modules_owned","adoption_pct","csat","expansion_score","recommended_action","expansion_value_proxy"]],use_container_width=True,hide_index=True)
    st.download_button("Download expansion candidates CSV",exp.to_csv(index=False).encode(),"expansion_candidates.csv","text/csv")

with tabs[5]:
    st.subheader("Data quality and scenario assumptions")
    fields=["account_id","account_name","contract_value","renewal_in_days","tickets_30d","tickets_prev_30d","sla_breaches_30d","escalations_30d","csat","milestones_slipped","adoption_pct"]
    quality=[]
    for c in fields:
        quality.append({"Field":c,"Status":"Missing field — defaults applied" if c not in raw.columns else ("OK" if raw[c].isna().sum()==0 else "Missing values"),"Missing values":len(raw) if c not in raw.columns else int(raw[c].isna().sum())})
    st.dataframe(pd.DataFrame(quality),use_container_width=True,hide_index=True)
    st.write(f"Rows: {len(raw):,} | Duplicate account IDs: {int(df.account_id.duplicated().sum())}")
    st.markdown(f"""
    **Current scenario assumptions**
    - Contribution margin: {margin:.0%}
    - Remaining-value horizon: {horizon_years} years
    - Assumed intervention success: {retention_success:.0%}
    - Assumed expansion win rate: {expansion_win:.0%}
    - High-risk threshold: {high_threshold}/100
    - Watch threshold: {watch_threshold}/100
    """)
    st.warning("These outputs are scenario estimates based on synthetic data and assumed weights/rates. They are not observed revenue, validated churn probabilities, or proof of causal impact.")
    st.caption("For validation, compare predictions with historical renewal/churn outcomes and actual intervention costs.")

st.divider()
st.caption("Prototype aligned to the Post-Sale Intelligence Gap problem analysis: cross-system signals → value-weighted prioritisation → decision/approval → action measurement. Live Zoho integration and outcome tracking remain to be implemented.")
