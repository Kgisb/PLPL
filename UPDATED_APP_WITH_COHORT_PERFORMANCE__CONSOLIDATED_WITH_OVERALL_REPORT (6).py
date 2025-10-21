
# ---- Compact Status Bar (badges) ----

def _render_status_bar(excluded_count: int, excluded_col: str, rows_in_scope: int, track_val: str):
    import streamlit as st
# === Kids detail renderer (clean, self-contained) ===
def _render_marketing_kids_detail(df):
    import pandas as _pd, numpy as _np, re as _re, streamlit as st
    if df is None or getattr(df, "empty", True):
        st.info("No data available."); return

    def _col_exists(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low: return low[c.lower()]
        for c in df.columns:
            for cand in candidates:
                if cand.lower() in c.lower(): return c
        return None

    def _dt_parse_dayfirst(series):
        s = series.astype(str).str.strip().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
        dt = _pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        need = dt.isna()
        if need.any():
            compact = s.where(need)
            mask = compact.str.fullmatch(r"\d{8}", na=False)
            dt2 = _pd.to_datetime(compact.where(mask), format="%d%m%Y", errors="coerce")
            dt = dt.fillna(dt2)
        need = dt.isna()
        if need.any():
            lead10 = s.where(need).str.slice(0,10)
            dt3 = _pd.to_datetime(lead10, errors="coerce", dayfirst=True)
            dt = dt.fillna(dt3)
        return dt

    deal_col = _col_exists(df, ["Deal Name","Deal name","Name","Deal","Title"])
    create_col = _col_exists(df, ["Create Date","Created Date","Deal Create Date","Date Created","Created On","Creation Date","Deal Created Date","Create_Date"])
    pay_col = _col_exists(df, ["Payment Received Date","Enrollment Date","Enrolment Date","Enrolled On","Payment Date","Payment_Received_Date"])
    trial_s_col = _col_exists(df, ["Trial Scheduled Date","Trial Schedule Date","Trial Booking Date","Trial Booked Date","Trial_Scheduled_Date"])
    trial_r_col = _col_exists(df, ["Trial Rescheduled Date","Trial Re-scheduled Date","Trial Reschedule Date","Trial_Rescheduled_Date"])
    calib_d_col = _col_exists(df, ["Calibration Done Date","Calibration Completed Date","First Calibration Done Date","Calibration Booking Date","Calibration Booked Date","First Calibration Scheduled Date"])
    source_col = _col_exists(df, ["JetLearn Deal Source","Deal Source","Original source","Source","Original traffic source"])

    if deal_col is None:
        st.error("Could not find the Deal Name column."); return

    create_dt = _dt_parse_dayfirst(df[create_col]) if create_col else _pd.Series(_pd.NaT, index=df.index)
    pay_dt = _dt_parse_dayfirst(df[pay_col]) if pay_col else _pd.Series(_pd.NaT, index=df.index)
    trial_s_dt = _dt_parse_dayfirst(df[trial_s_col]) if trial_s_col else _pd.Series(_pd.NaT, index=df.index)
    trial_r_dt = _dt_parse_dayfirst(df[trial_r_col]) if trial_r_col else _pd.Series(_pd.NaT, index=df.index)
    calib_d_dt = _dt_parse_dayfirst(df[calib_d_col]) if calib_d_col else _pd.Series(_pd.NaT, index=df.index)

    c1, c2, c3, c4 = st.columns(4)
    with c1: mode = st.selectbox("Mode", ["Entity", "Cohort"], index=0, key="mk_kids_mode")
    with c2:
        dflt_start = (create_dt.min() if create_dt.notna().any() else _pd.Timestamp("2023-01-01"))
        start = st.date_input("Start date", value=(dflt_start.date() if _pd.notna(dflt_start) else _pd.Timestamp("2023-01-01").date()), key="mk_kids_start")
    with c3:
        dflt_end = (create_dt.max() if create_dt.notna().any() else _pd.Timestamp.today())
        end = st.date_input("End date", value=(dflt_end.date() if _pd.notna(dflt_end) else _pd.Timestamp.today().date()), key="mk_kids_end")
    with c4: only_organic = st.checkbox("Only Organic?", value=True, key="mk_kids_only_org")

    # broader 's kid' variants: optional apostrophe, flexible separators
    name_pat = _re.compile(r"(?:^|\b)[’']?\s*s\s*[-_./]*\s*kid\b", flags=_re.IGNORECASE)
    is_kids = df[deal_col].astype(str).str.contains(name_pat)

    # refine by exact names (multiselect)
    _candidates = sorted(df.loc[is_kids, deal_col].astype(str).unique().tolist())
    _selected = st.multiselect("Names (auto-matched)", options=_candidates, default=_candidates, key="mk_kids_names")
    if _selected:
        is_kids = is_kids & df[deal_col].astype(str).isin(_selected)
# removed stray is_org line

    start_ts = _pd.Timestamp(start); end_ts = _pd.Timestamp(end) + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1)
    def in_range(series): return (series >= start_ts) & (series <= end_ts)

    if mode == "Entity":
        f_create = in_range(create_dt); f_pay = in_range(pay_dt)
        f_trial_s = in_range(trial_s_dt); f_trial_r = in_range(trial_r_dt); f_calib_d = in_range(calib_d_dt)
    else:
        base = in_range(create_dt); f_create=f_pay=f_trial_s=f_trial_r=f_calib_d=base

    base_all = (is_org if only_organic else _pd.Series(True, index=df.index)) & (create_dt.notna())
    base_kids = base_all & is_kids

    nuniq = lambda m: df.loc[m, deal_col].nunique()
    cnt_created_kids = nuniq(base_kids & f_create)
    cnt_trial_s_kids = nuniq(base_kids & f_trial_s)
    cnt_trial_r_kids = nuniq(base_kids & f_trial_r)
    cnt_calib_d_kids = nuniq(base_kids & f_calib_d)
    cnt_enroll_kids = nuniq(base_kids & f_pay)

    if source_col:
        base_org = (df[source_col].astype(str).str.lower().str.contains("organic")) & (create_dt.notna())
    else:
        base_org = _pd.Series(False, index=df.index)

    denom_org = nuniq(base_org & f_create) if base_org.any() else 0
    denom_all = nuniq((create_dt.notna()) & f_create)
    pct = lambda a,b: (a/b*100.0) if b else 0.0

    st.markdown("### Funnel – Kids Deals (matching **“s kid”**)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Created (Kids)", cnt_created_kids, delta=f"{pct(cnt_created_kids, denom_org):.1f}% of Organic | {pct(cnt_created_kids, denom_all):.1f}% of All")
    m2.metric("Trial Scheduled (Kids)", cnt_trial_s_kids)
    m3.metric("Trial Rescheduled (Kids)", cnt_trial_r_kids)
    m4.metric("Calibration Done (Kids)", cnt_calib_d_kids)
    m5.metric("Enrollments (Kids)", cnt_enroll_kids)

    table = _pd.DataFrame({
        "Stage": ["Created","Trial Scheduled","Trial Rescheduled","Calibration Done","Enrollments"],
        "Count (Kids)": [cnt_created_kids, cnt_trial_s_kids, cnt_trial_r_kids, cnt_calib_d_kids, cnt_enroll_kids],
        "% of Organic": [pct(cnt_created_kids, denom_org), _np.nan, _np.nan, _np.nan, _np.nan],
        "% of All": [pct(cnt_created_kids, denom_all), _np.nan, _np.nan, _np.nan, _np.nan],
    })
    st.dataframe(table, use_container_width=True)
    st.download_button("Download table (CSV) — Kids detail", data=table.to_csv(index=False).encode("utf-8"),
                       file_name="kids_detail_funnel.csv", mime="text/csv", key="dl_mk_kids_table_final")
# === /Kids detail renderer ===

    html = f"""
    <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:2px 0 4px;">
      <span style="font-size:12px; color:#64748B;">
        <span style="opacity:.85">Excluded</span>
        <span>“1.2 Invalid deal(s)”</span>
        <span style="opacity:.6">·</span>
        <span>{excluded_count:,} rows</span>
        <span style="opacity:.55">({excluded_col})</span>
      </span>
      <span style="font-size:12px; color:#64748B;">
        <span style="opacity:.85">In scope</span>
        <span>{rows_in_scope:,}</span>
      </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
# ---- Global Refresh (reset filters & cache) ----
def _reset_all_filters_and_cache(preserve_nav=True):
    import streamlit as st
    for clear_fn in (
        getattr(getattr(st, "cache_data", object()), "clear", None),
        getattr(getattr(st, "cache_resource", object()), "clear", None),
        getattr(getattr(st, "experimental_memo", object()), "clear", None),
        getattr(getattr(st, "experimental_singleton", object()), "clear", None)
    ):
        try:
            if callable(clear_fn):
                clear_fn()
        except Exception:
            pass
    keep_keys = set()
    if preserve_nav:
        keep_keys |= {"nav_master", "nav_sub", "nav_master_prev"}
    rm_tags = [
        "filter", "selected", "select", "multiselect", "radio", "checkbox",
        "date", "from", "to", "range", "track", "cohort", "pareto",
        "country", "state", "city", "source", "deal", "stage",
        "owner", "counsellor", "counselor", "team",
        "segment", "sku", "plan", "product",
        "data_src_input"
    ]
    to_delete = []
    for k in list(st.session_state.keys()):
        if k in keep_keys:
            continue
        kl = k.lower()
        if any(tag in kl for tag in rm_tags):
            to_delete.append(k)
    for k in to_delete:
        try:
            del st.session_state[k]
        except Exception:
            pass
# ---- Global CSS polish (no logic change) ----
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px !important; padding-top: 1.2rem !important; padding-bottom: 2.0rem !important; }
        .stAltairChart, .stPlotlyChart, .stVegaLiteChart, .stDataFrame, .stTable, .element-container [data-baseweb="table"] {
            border: 1px solid #e7e8ea; border-radius: 16px; padding: 14px; background: #ffffff; box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
        }
        .stDataFrame [role="grid"] { border-radius: 12px; overflow: hidden; border: 1px solid #e7e8ea; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        div[data-testid="stMetric"] { background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%); border: 1px solid #eef0f2; border-radius: 16px; padding: 14px 16px; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        details[data-testid="stExpander"] { border: 1px solid #e7e8ea; border-radius: 14px; background: #ffffff; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        details[data-testid="stExpander"] summary { font-weight: 600; color: #0f172a; }
        button[role="tab"] { border-radius: 999px !important; padding: 8px 14px !important; margin-right: 6px !important; border: 1px solid #e7e8ea !important; }
        button[role="tab"][aria-selected="true"] { background: #111827 !important; color: #ffffff !important; border-color: #111827 !important; }
        div[data-baseweb="select"], .stTextInput > div, .stNumberInput > div, .stDateInput > div { border-radius: 12px !important; box-shadow: 0 1px 4px rgba(16,24,40,.04); }
        .stSlider > div { padding-top: 10px; }
        .stButton > button, .stDownloadButton > button { border-radius: 12px !important; border: 1px solid #11182720 !important; box-shadow: 0 2px 8px rgba(16,24,40,.08) !important; transition: transform .05s ease-in-out; }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0f172a; letter-spacing: 0.1px; }
        .stMarkdown hr { margin: 18px 0; border: none; border-top: 1px dashed #d6d8db; }
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception:
    pass

# app.py — JetLearn: MIS + Predictibility + Trend & Analysis + 80-20 (Merged, de-conflicted)

import streamlit as st




# ======================
# Performance ▶ Leaderboard  (with "All" rows + Overall circular totals)
# ======================

def _render_performance_comparison(
    df_f, create_col, pay_col, counsellor_col, country_col, source_col,
    first_cal_sched_col=None, cal_resched_col=None, cal_done_col=None
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    from datetime import date

    st.subheader("Performance — Comparison (Window A vs Window B)")
    st.caption("Compare metrics across two independently-filtered windows (A & B) with separate date ranges. "
               "MTD = Payment in window AND Created in window; Cohort = Payment in window (create anywhere).")

    # ---- safe column resolver
    def _col(df, primary, cands):
        if primary and primary in df.columns: return primary
        for c in cands:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None

    _create = _col(df_f, create_col, ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _col(df_f, pay_col,    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _owner  = _col(df_f, counsellor_col, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
    _cntry  = _col(df_f, country_col,    ["Country","Country Name"])
    _src    = _col(df_f, source_col,     ["JetLearn Deal Source","Deal Source","Source"])

    if _create is None or _pay is None:
        st.error("Required date columns not found (Create Date / Payment Received Date).")
        return

    # ---- datetimes + safe strings
    d = df_f.copy()
    def _to_dt(s):
        if pd.api.types.is_datetime64_any_dtype(s): return s
        try:    return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except: return pd.to_datetime(s, errors="coerce")

    d["_create"] = _to_dt(d[_create])
    d["_pay"]    = _to_dt(d[_pay])
    if _owner: d["_owner"] = d[_owner].fillna("Unknown").astype(str)
    if _cntry: d["_cntry"] = d[_cntry].fillna("Unknown").astype(str)
    if _src:   d["_src"]   = d[_src].fillna("Unknown").astype(str)

    # Optional calibration dates
    _first  = _col(d, first_cal_sched_col, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
    _resch  = _col(d, cal_resched_col,     ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
    _done   = _col(d, cal_done_col,        ["Calibration Done Date","Cal Done","Trial Done Date"])
    def _to_dt2(colname):
        if not colname or colname not in d.columns: return None
        try:    return pd.to_datetime(d[colname], dayfirst=True, errors="coerce")
        except: return pd.to_datetime(d[colname], errors="coerce")
    d["_first"] = _to_dt2(_first)
    d["_resch"] = _to_dt2(_resch)
    d["_done"]  = _to_dt2(_done)

    # ---- controls
    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="cmp_mode")
    metrics = st.multiselect(
        "Metrics to compare",
        ["Deals Created","Enrollments","First Cal","Cal Rescheduled","Cal Done"],
        default=["Enrollments","Deals Created"],
        key="cmp_metrics",
    )
    if not metrics:
        st.info("Select at least one metric to compare."); return

    dim_opts = []
    if _owner: dim_opts.append("Academic Counsellor")
    if _src:   dim_opts.append("JetLearn Deal Source")
    if _cntry: dim_opts.append("Country")
    if not dim_opts:
        st.warning("No grouping dimensions available."); return

    st.markdown("**Configure Windows**")
    colA, colB = st.columns(2)
    today = date.today()
    with colA:
        st.write("### Window A")
        dims_a = st.multiselect("Group by (A)", dim_opts, default=[dim_opts[0]], key="cmp_dims_a")
        date_a = st.date_input("Date range (A)", value=(today.replace(day=1), today), key="cmp_date_a")
    with colB:
        st.write("### Window B")
        dims_b = st.multiselect("Group by (B)", dim_opts, default=[dim_opts[0]], key="cmp_dims_b")
        date_b = st.date_input("Date range (B)", value=(today.replace(day=1), today), key="cmp_date_b")

    def _ensure_tuple(val):
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return pd.to_datetime(val[0]), pd.to_datetime(val[1])
        return pd.to_datetime(val), pd.to_datetime(val)

    a_start, a_end = _ensure_tuple(date_a)
    b_start, b_end = _ensure_tuple(date_b)
    if a_end < a_start: a_start, a_end = a_end, a_start
    if b_end < b_start: b_start, b_end = b_end, b_start

    def _agg(df, dims, start, end):
        # map UI dims to internal cols
        group_cols = []
        if dims:
            for dname in dims:
                if dname == "Academic Counsellor" and "_owner" in df: group_cols.append("_owner")
                if dname == "JetLearn Deal Source" and "_src" in df: group_cols.append("_src")
                if dname == "Country" and "_cntry" in df: group_cols.append("_cntry")
        if not group_cols:
            df = df.copy(); df["_dummy"] = "All"; group_cols = ["_dummy"]

        g = df.copy()
        m_create = g["_create"].between(start, end)
        m_pay    = g["_pay"].between(start, end)
        m_first  = g["_first"].between(start, end) if "_first" in g else pd.Series(False, index=g.index)
        m_resch  = g["_resch"].between(start, end) if "_resch" in g else pd.Series(False, index=g.index)
        m_done   = g["_done"].between(start, end)  if "_done"  in g else pd.Series(False, index=g.index)

        res = g[group_cols].copy()

        for m in metrics:
            if m == "Deals Created":
                cnt = g.loc[m_create].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Deals Created"), left_on=group_cols, right_index=True, how="left")
            elif m == "Enrollments":
                if mode == "MTD":
                    cnt = g.loc[m_pay & m_create].groupby(group_cols, dropna=False).size()
                else:
                    cnt = g.loc[m_pay].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Enrollments"), left_on=group_cols, right_index=True, how="left")
            elif m == "First Cal":
                if mode == "MTD":
                    cnt = g.loc[m_first & m_create].groupby(group_cols, dropna=False).size()
                else:
                    cnt = g.loc[m_first].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("First Cal"), left_on=group_cols, right_index=True, how="left")
            elif m == "Cal Rescheduled":
                if mode == "MTD":
                    cnt = g.loc[m_resch & m_create].groupby(group_cols, dropna=False).size()
                else:
                    cnt = g.loc[m_resch].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Cal Rescheduled"), left_on=group_cols, right_index=True, how="left")
            elif m == "Cal Done":
                if mode == "MTD":
                    cnt = g.loc[m_done & m_create].groupby(group_cols, dropna=False).size()
                else:
                    cnt = g.loc[m_done].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Cal Done"), left_on=group_cols, right_index=True, how="left")

        res = res.groupby(group_cols, dropna=False).first().fillna(0).reset_index()
        pretty = []
        for c in group_cols:
            if c == "_owner": pretty.append("Academic Counsellor")
            elif c == "_src": pretty.append("JetLearn Deal Source")
            elif c == "_cntry": pretty.append("Country")
            elif c == "_dummy": pretty.append("All")
            else: pretty.append(c)
        res = res.rename(columns=dict(zip(group_cols, pretty)))
        return res, pretty

    res_a, _ = _agg(d, dims_a, a_start, a_end)
    res_b, _ = _agg(d, dims_b, b_start, b_end)

    # Join results
    join_keys = [c for c in ["Academic Counsellor","JetLearn Deal Source","Country","All"] if c in res_a.columns and c in res_b.columns]
    if join_keys:
        merged = pd.merge(res_a, res_b, on=join_keys, how="outer", suffixes=(" (A)", " (B)"))
    else:
        def _label(df):
            keys = [k for k in ["Academic Counsellor","JetLearn Deal Source","Country","All"] if k in df.columns]
            if keys: return df.assign(_KeyLabel=df[keys].astype(str).agg(" | ".join, axis=1))
            return df.assign(_KeyLabel="All")
        ra = _label(res_a); rb = _label(res_b)
        merged = pd.merge(ra, rb, on="_KeyLabel", how="outer", suffixes=(" (A)", " (B)"))

    # Deltas (numeric-safe)
    for m in metrics:
        colA = f"{m} (A)"; colB = f"{m} (B)"
        if colA in merged.columns and colB in merged.columns:
            a = pd.to_numeric(merged[colA], errors="coerce").fillna(0.0)
            b = pd.to_numeric(merged[colB], errors="coerce").fillna(0.0)
            merged[f"Δ {m} (B−A)"] = (b - a)
            denom = a.to_numpy(); num = b.to_numpy()
            pct_arr = np.where(denom != 0, (num / denom) * 100.0, np.nan)
            merged[f"% Δ {m} (vs A)"] = pd.Series(pct_arr, index=merged.index).round(1)

    # Display
    key_cols = [c for c in ["Academic Counsellor","JetLearn Deal Source","Country","All"] if c in merged.columns]
    a_cols = [f"{m} (A)" for m in metrics if f"{m} (A)" in merged.columns]
    b_cols = [f"{m} (B)" for m in metrics if f"{m} (B)" in merged.columns]
    d_cols = [c for c in merged.columns if c.startswith("Δ ") or c.startswith("% Δ ")]

    final_cols = key_cols + a_cols + b_cols + d_cols
    final = merged[final_cols].fillna(0)

    # ---- Optional Overall row + Top-N limiter ----
    st.divider()
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        show_overall = st.checkbox("Show Overall row", value=True, key="cmp_show_overall")
    rank_options = [c for c in (a_cols + b_cols + d_cols) if c in final.columns]
    default_rank = next((pref for pref in [f"% Δ Enrollments (vs A)", f"Enrollments (B)", f"Deals Created (B)"] if pref in rank_options), (rank_options[0] if rank_options else None))
    with c2:
        limit_choice = st.selectbox("Limit rows", ["All","Top 10","Top 15"], index=0, key="cmp_limit_rows")
    with c3:
        rank_by = st.selectbox("Rank by", rank_options or ["(none)"], index=(0 if rank_options else 0), key="cmp_rank_by")

    to_show = final.copy()
    if rank_options and rank_by in to_show.columns:
        _sort_vals = pd.to_numeric(to_show[rank_by], errors="coerce")
        to_show = to_show.assign(_sort=_sort_vals.fillna(float("-inf"))).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    if limit_choice == "Top 10": to_show = to_show.head(10)
    elif limit_choice == "Top 15": to_show = to_show.head(15)

    if show_overall and not to_show.empty:
        num_cols = [c for c in to_show.columns if c not in key_cols]
        overall = {k: "Overall" for k in key_cols}
        sums = to_show[num_cols].apply(pd.to_numeric, errors="coerce").sum(numeric_only=True)
        overall.update({c: sums.get(c, 0.0) for c in num_cols})
        to_show = pd.concat([to_show, pd.DataFrame([overall])], ignore_index=True)

        # Recompute %Δ for Overall from summed A/B
        for m in metrics:
            ca = f"{m} (A)"; cb = f"{m} (B)"; cp = f"% Δ {m} (vs A)"
            if ca in to_show.columns and cb in to_show.columns and cp in to_show.columns:
                a_sum = pd.to_numeric(to_show.loc[to_show.index[-1], ca], errors="coerce")
                b_sum = pd.to_numeric(to_show.loc[to_show.index[-1], cb], errors="coerce")
                pct = (b_sum / a_sum * 100.0) if (pd.notna(a_sum) and a_sum != 0) else np.nan
                to_show.loc[to_show.index[-1], cp] = round(pct, 1) if pd.notna(pct) else np.nan

    st.dataframe(to_show, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Comparison (A vs B)",
        to_show.to_csv(index=False).encode("utf-8"),
        file_name="performance_comparison_A_vs_B.csv",
        mime="text/csv"
    )


def _render_performance_leaderboard(
    df_f,
    counsellor_col,
    create_col,
    pay_col,
    first_cal_sched_col,
    cal_resched_col,
    cal_done_col,
    source_col,
    ref_intent_col,
):
    st.subheader("Performance — Leaderboard (Academic Counsellor)")

    if not counsellor_col or counsellor_col not in df_f.columns:
        st.warning("Academic Counsellor column not found.", icon="⚠️")
        return

    # Date mode and scope
    level = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="lb_mode")
    date_mode = st.radio("Date scope", ["This month", "Last month", "Custom"], index=0, horizontal=True, key="lb_scope")

    today = date.today()
    def _month_bounds(d: date):
        from calendar import monthrange
        start = date(d.year, d.month, 1)
        end = date(d.year, d.month, monthrange(d.year, d.month)[1])
        return start, end
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if date_mode == "This month":
        range_start, range_end = _month_bounds(today)
    elif date_mode == "Last month":
        range_start, range_end = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: range_start = st.date_input("Start", value=today.replace(day=1), key="lb_start")
        with c2: range_end   = st.date_input("End",   value=_month_bounds(today)[1], key="lb_end")
        if range_end < range_start:
            st.error("End date cannot be before start date.")
            return

    # --- Normalize fields ---
    def _dt(s):
        try:
            return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True).dt.date
        except Exception:
            return pd.to_datetime(pd.Series([None]*len(s)), errors="coerce").dt.date

    _C = _dt(df_f[create_col]) if (create_col and create_col in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    _P = _dt(df_f[pay_col])    if (pay_col and pay_col in df_f.columns)     else pd.Series(pd.NaT, index=df_f.index)
    _F = _dt(df_f[first_cal_sched_col]) if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    _R = _dt(df_f[cal_resched_col])     if (cal_resched_col and cal_resched_col in df_f.columns)         else pd.Series(pd.NaT, index=df_f.index)
    _D = _dt(df_f[cal_done_col])        if (cal_done_col and cal_done_col in df_f.columns)              else pd.Series(pd.NaT, index=df_f.index)

    _SRC  = df_f[source_col].fillna("Unknown").astype(str).str.strip() if (source_col and source_col in df_f.columns) else pd.Series("Unknown", index=df_f.index)
    _REFI = df_f[ref_intent_col].fillna("Unknown").astype(str).str.strip() if (ref_intent_col and ref_intent_col in df_f.columns) else pd.Series("Unknown", index=df_f.index)

    # --- Window masks ---
    c_in = _C.between(range_start, range_end)
    p_in = _P.between(range_start, range_end)
    f_in = _F.between(range_start, range_end)
    r_in = _R.between(range_start, range_end)
    d_in = _D.between(range_start, range_end)

    # Mode rules
    if level == "MTD":
        enrol_mask = p_in & c_in
        f_mask = f_in & c_in
        r_mask = r_in & c_in
        d_mask = d_in & c_in
        referral_created_mask = c_in & _SRC.str.contains("referr", case=False, na=False)
        sales_generated_intent_mask = c_in & _REFI.str.contains(r"\bsales\s*generated\b", case=False, na=False, regex=True)
    else:
        enrol_mask = p_in
        f_mask = f_in
        r_mask = r_in
        d_mask = d_in
        referral_created_mask = _SRC.str.contains("referr", case=False, na=False) & c_in
        sales_generated_intent_mask = _REFI.str.contains(r"\bsales\s*generated\b", case=False, na=False, regex=True) & c_in
    sales_intent_enrol_mask = enrol_mask & _REFI.str.contains(r"\bsales\s*generated\b", case=False, na=False, regex=True)

    # Referral Enrolments: enrolments where Deal Source indicates Referrals
    referral_enrol_mask = enrol_mask & _SRC.str.contains("referr", case=False, na=False)

    # Deals count is based on Create Date
    deals_mask = c_in

    # Group & aggregate
    grp = df_f[counsellor_col].fillna("Unknown").astype(str)
    out = pd.DataFrame({
        "Academic Counsellor": grp,
        "Deals": deals_mask.astype(int),
        "Enrolments": enrol_mask.astype(int),
        "First Cal": f_mask.astype(int),
        "Cal Rescheduled": r_mask.astype(int),
        "Cal Done": d_mask.astype(int),
        "Referral Deals (Deal Source=Referrals)": referral_created_mask.astype(int),
        "Referral Intent (Sales generated)": sales_generated_intent_mask.astype(int),        "Sales Intent Enrolments": sales_intent_enrol_mask.astype(int),

        "Referral Enrolments": referral_enrol_mask.astype(int),})
    tbl = out.groupby("Academic Counsellor").sum(numeric_only=True).reset_index()

    # ---- Overall totals (across current filters & date window) ----
    totals = {
        "Deals":              int(tbl["Deals"].sum()) if "Deals" in tbl.columns else 0,
        "Enrolments":         int(tbl["Enrolments"].sum()) if "Enrolments" in tbl.columns else 0,
        "First Cal":          int(tbl["First Cal"].sum()) if "First Cal" in tbl.columns else 0,
        "Cal Rescheduled":    int(tbl["Cal Rescheduled"].sum()) if "Cal Rescheduled" in tbl.columns else 0,
        "Cal Done":           int(tbl["Cal Done"].sum()) if "Cal Done" in tbl.columns else 0,
        "Referral Deals (Deal Source=Referrals)": int(tbl["Referral Deals (Deal Source=Referrals)"].sum()) if "Referral Deals (Deal Source=Referrals)" in tbl.columns else 0,
        "Referral Intent (Sales generated)":      int(tbl["Referral Intent (Sales generated)"].sum()) if "Referral Intent (Sales generated)" in tbl.columns else 0,        "Sales Intent Enrolments": int(tbl["Sales Intent Enrolments"].sum()) if "Sales Intent Enrolments" in tbl.columns else 0,

        "Referral Enrolments": int(tbl["Referral Enrolments"].sum()) if "Referral Enrolments" in tbl.columns else 0,        "Referral Enrolments": int(tbl["Referral Enrolments"].sum()) if "Referral Enrolments" in tbl.columns else 0,
    }

    # Ranking controls
    metric = st.selectbox(
        "Rank by",
        ["Enrolments","Deals","First Cal","Cal Rescheduled","Cal Done","Referral Deals (Deal Source=Referrals)","Referral Intent (Sales generated)", "Sales Intent Enrolments", "Referral Enrolments"],
        index=0,
        key="lb_rank_metric"
    )
    ascending = st.checkbox("Ascending order", value=False, key="lb_asc")
    tbl = tbl.sort_values(metric, ascending=ascending).reset_index(drop=True)
    tbl.index = tbl.index + 1

    # ---- Overall (circular badges) ----
    overall_html = r"""
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:.25rem 0 1rem 0'>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Deals</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{deals}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Enrolments</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{enrol}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>First Cal</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{fcal}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Cal Rescheduled</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{rres}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Cal Done</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{cdone}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Referral Deals</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{refd}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Referral Enrolments</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{refenrol}</span>
          </div>

          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Ref Intent (Sales gen)</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{refi}</span>
          </div>
          <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:9999px; background:#F1F5F9; border:1px solid #E5E7EB;'>
            <span style='font-weight:700;'>Sales Intent Enrolments</span>
            <span style='width:44px;height:44px;border-radius:9999px;display:flex;align-items:center;justify-content:center;border:3px solid #CBD5E1;font-weight:800;'>{sienrol}</span>
          </div>
        </div>
    """
    with st.container():
        st.markdown("### Overall")
        st.markdown(
            overall_html.format(
                refenrol=totals.get("Referral Enrolments", 0),
                sienrol=totals.get("Sales Intent Enrolments", 0),
                deals=totals.get("Deals", 0),
                enrol=totals.get("Enrolments", 0),
                fcal=totals.get("First Cal", 0),
                rres=totals.get("Cal Rescheduled", 0),
                cdone=totals.get("Cal Done", 0),
                refd=totals.get("Referral Deals (Deal Source=Referrals)", 0),
                refi=totals.get("Referral Intent (Sales generated)", 0),
            ),
            unsafe_allow_html=True
        )

    # Rows control (Top 10 / Top 25 / All)
    
    # Rows control (Top 10 / Top 25 / All) + Overall row toggle
    show_n = st.radio('Rows', ['Top 10','Top 25','All'], index=2, horizontal=True, key='lb_rows')
    include_overall = st.toggle("Add 'Overall' as first row", value=False, key='lb_overall_row')

    # Build display table with optional Overall on top
    tbl_display = tbl.copy()
    if show_n != 'All':
        n = 10 if show_n == 'Top 10' else 25
        tbl_display = tbl_display.head(n)

    if include_overall:
        overall_row = pd.DataFrame([{
            "Academic Counsellor": "Overall",
            "Deals": totals.get("Deals", 0),
            "Enrolments": totals.get("Enrolments", 0),
            "First Cal": totals.get("First Cal", 0),
            "Cal Rescheduled": totals.get("Cal Rescheduled", 0),
            "Cal Done": totals.get("Cal Done", 0),
            "Referral Deals (Deal Source=Referrals)": totals.get("Referral Deals (Deal Source=Referrals)", 0),
            "Referral Intent (Sales generated)": totals.get("Referral Intent (Sales generated)", 0),
        }])
        # Always keep Overall at the top even when limiting rows
        if show_n != 'All':
            # Keep top (n-1) counsellor rows to make room for Overall
            n = 10 if show_n == 'Top 10' else 25
            top_rows = tbl.head(max(n-1, 0))
            tbl_display = pd.concat([overall_row, top_rows], ignore_index=True)
        else:
            tbl_display = pd.concat([overall_row, tbl_display], ignore_index=True)

    # Pretty index starting at 1
    tbl_display.index = range(1, len(tbl_display)+1)
    st.dataframe(tbl_display, use_container_width=True)

    st.caption(f"Window: **{range_start} → {range_end}** • Mode: **{level}**")

# --- Safe default for optional UI text blobs ---
try:
    DATA_SOURCE_TEXT
except NameError:
    DATA_SOURCE_TEXT = ""
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re
from datetime import date, timedelta


# ======================
# Page & minimal styling
# ======================
st.set_page_config(page_title="JetLearn – MIS + Trend + 80-20", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .stAltairChart {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
      }
      .legend-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
      }
      .pill-total { background: #e5e7eb; }
      .pill-ai    { background: #bfdbfe; }
      .pill-math  { background: #bbf7d0; }

      .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
      }
      .kpi-title { color:#6b7280; font-size:.9rem; margin-bottom:6px; }
      .kpi-value { font-weight:700; font-size:1.4rem; color:#111827; }
      .kpi-sub   { color:#6b7280; font-size:.85rem; }
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .25rem;
      }
      .chip {
        display:inline-block; padding:4px 8px; border-radius:999px;
        background:#f3f4f6; color:#374151; font-size:.8rem; margin-top:.25rem;
      }
      .muted { color:#6b7280; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = {
    "Total": "#6b7280",
    "AI Coding": "#2563eb",
    "Math": "#16a34a",
    "ThresholdLow": "#f3f4f6",
    "ThresholdMid": "#e5e7eb",
    "ThresholdHigh": "#d1d5db",
    "A_actual": "#2563eb",
    "Rem_prev": "#6b7280",
    "Rem_same": "#16a34a",
}

# ======================
# Helpers (shared)
# ======================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=series.index if series is not None else None)
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if s.notna().sum() == 0:
        for unit in ["s", "ms"]:
            try:
                s = pd.to_datetime(series, errors="coerce", unit=unit)
                break
            except Exception:
                pass
    return s

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

# Invalid deals exclusion
INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", flags=re.IGNORECASE)
def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    mask_keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v: return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

# Key-source mapping (Referral / PM buckets)
def normalize_key_source(val: str) -> str:
    if not isinstance(val, str): return "Other"
    v = val.strip().lower()
    if "referr" in v: return "Referral"
    if "pm" in v and "search" in v: return "PM - Search"
    if "pm" in v and "social" in v: return "PM - Social"
    return "Other"

def assign_src_pick(df: pd.DataFrame, source_col: str | None, use_key: bool) -> pd.DataFrame:
    d = df.copy()
    if source_col and source_col in d.columns:
        if use_key:
            d["_src_pick"] = d[source_col].apply(normalize_key_source)
        else:
            d["_src_pick"] = d[source_col].fillna("Unknown").astype(str)
    else:
        d["_src_pick"] = "Other"
    return d

# ======================
# Load data & global sidebar
# ======================
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"  # point to /mnt/data/Master_sheet-DB.csv if needed

if "data_src" not in st.session_state:
    st.session_state["data_src"] = DEFAULT_DATA_PATH

def _update_data_src():
    import streamlit as st
    DEFAULT = globals().get('DEFAULT_DATA_PATH', 'Master_sheet-DB.csv')
    st.session_state['data_src'] = st.session_state.get('data_src_input', DEFAULT)
    try:
        st.rerun()
    except Exception:
        pass
    import streamlit as st
    DEFAULT = globals().get('DEFAULT_DATA_PATH', 'Master_sheet-DB.csv')
    st.session_state['data_src'] = st.session_state.get('data_src_input', DEFAULT)
    try:
        st.rerun()
    except Exception:
        pass

with st.sidebar:
    st.header("JetLearn • Navigation")
    # Master tabs -> Sub tabs (contextual)
    MASTER_SECTIONS = {
        "Performance": ["Cash-in", "Dashboard", "MIS", "Daily Business", "Sales Tracker", "AC Wise Detail", "Leaderboard", "Quick View", "Comparison", "Sales Activity", "Deal stage", "Original source", "Referral / No-Referral", "Lead mix", "Referral performance", "Slow Working Deals", "Activity concentration"],
        "Funnel & Movement": ["Funnel", "Lead Movement", "Stuck deals", "Deal Velocity", "Deal Decay", "Carry Forward", "Referral Pitched In", "Closed Lost Analysis"],
        "Insights & Forecast": ["Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph"],
        "Marketing": ["Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement","Kids detail", "Deal Detail", "Sales Intern Funnel", "Master analysis", "Referral Tracking", "Talk Time", "Overall Report"],
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    # Replace sidebar 'View' radio with session wiring (UI moves to main area)
    sub_views = MASTER_SECTIONS.get(master, [])
    if 'nav_sub' not in st.session_state or st.session_state.get('nav_master_prev') != master:
        st.session_state['nav_sub'] = sub_views[0] if sub_views else ''
    st.session_state['nav_master_prev'] = master
    sub = st.session_state['nav_sub']
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)
    st.caption("Use MIS for status; Predictibility for forecast; Trend & Analysis for grouped drilldowns; 80-20 for Pareto & Mix.")


    st.markdown("<div style=\"height:6px\"></div>", unsafe_allow_html=True)
    st.markdown("<div style=\"height:4px\"></div>", unsafe_allow_html=True)
    try:
        _trk = track if 'track' in locals() else st.session_state.get('track', '')
        if _trk:
            st.caption(f"<span data-testid=\"track-caption-bottom\">Track: <strong>{_trk}</strong></span>", unsafe_allow_html=True)
    except Exception:
        pass
def _render_marketing_overall_report(df_f: pd.DataFrame):
    import pandas as pd
    import numpy as np
    import streamlit as st
    from datetime import date, timedelta
    
    # --- Column resolution ---
    def _col(df, names):
        
        # Prefer _col_exists if available, else fall back to find_col, else simple scan
        if '_col_exists' in globals():
            return _col_exists(df, names)
        if 'find_col' in globals():
            return find_col(df, names)
        # manual fallback
        for c in names:
            if c in df.columns: 
                return c
        low = {c.lower(): c for c in df.columns}
        for c in names:
            if c.lower() in low:
                return low[c.lower()]
        # loose contains match
        for c in df.columns:
            for cand in names:
                if cand.lower() in c.lower():
                    return c
        return None