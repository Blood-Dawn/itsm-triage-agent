"""
app/streamlit_app.py
────────────────────
M8 — Streamlit demo UI for the ITSM Triage Agent.

This app calls the FastAPI server (M4) over HTTP. Run the server first:

    uvicorn api.app:app --reload

Then launch the UI:

    streamlit run app/streamlit_app.py

Or with Docker Compose (both services together):

    docker compose up

HOW IT WORKS:

    The app is a thin HTTP client. It sends the ticket text to
    POST /triage and displays the structured response. All model
    logic lives in the API server — the UI has zero ML imports.

    This separation is intentional: the frontend is decoupled from
    the model, so you can swap backends, change the model, or deploy
    the UI separately without touching this file.

FEATURES:
    - Sample ticket picker (8 categories) for quick demo
    - Backend selector: finetuned | baseline | compare both
    - Color-coded category and priority badges
    - Confidence bars (finetuned backend)
    - LLM reasoning expander (baseline backend)
    - Side-by-side comparison mode
    - Server health status in sidebar
    - Latency display per request
"""

import os
import time

import requests
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ITSM Triage Agent",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# When running via docker compose, API_URL points to the api service internally.
# When running locally, it falls back to localhost:8000.
DEFAULT_SERVER_URL = os.environ.get("API_URL", "http://localhost:8000")

CATEGORY_COLORS = {
    "Network":     "#3B82F6",   # blue
    "Hardware":    "#F59E0B",   # amber
    "Software":    "#8B5CF6",   # purple
    "Security":    "#EF4444",   # red
    "Access":      "#10B981",   # emerald
    "Email":       "#06B6D4",   # cyan
    "Printer":     "#84CC16",   # lime
    "Performance": "#F97316",   # orange
}

PRIORITY_CONFIG = {
    "Critical": {"color": "#EF4444", "icon": "🔴", "border": "#FCA5A5"},
    "High":     {"color": "#F97316", "icon": "🟠", "border": "#FED7AA"},
    "Medium":   {"color": "#EAB308", "icon": "🟡", "border": "#FDE68A"},
    "Low":      {"color": "#22C55E", "icon": "🟢", "border": "#BBF7D0"},
}

SAMPLE_TICKETS = {
    "── Select a sample ticket ──": ("", ""),
    "🌐  Network — VPN outage": (
        "VPN not connecting — multiple users affected",
        "Users in the Miami office cannot connect to the corporate VPN since 8 AM. "
        "Tried restarting the VPN client and clearing credentials. Still getting "
        "a timeout error on connection. At least 15 users are affected and cannot "
        "access internal systems remotely.",
    ),
    "💻  Hardware — Laptop screen flickering": (
        "Laptop display flickering after driver update",
        "My laptop screen has been flickering on and off since I updated the display "
        "drivers yesterday morning. The flickering happens every 10-15 seconds and "
        "makes the laptop unusable. Rolling back the driver did not fix the issue.",
    ),
    "⚙️  Software — Excel crashing": (
        "Microsoft Excel crashes immediately on open",
        "Excel crashes every time I try to open any .xlsx file. The error message "
        "says 'Excel has stopped working'. I tried repairing the Office installation "
        "and reinstalling from scratch but the problem persists. This is blocking "
        "my ability to work on the Q4 budget.",
    ),
    "🔒  Security — Suspicious login alerts": (
        "Suspicious login attempts from unknown IP",
        "I received three security alerts this morning about failed login attempts "
        "on my account from an IP address in Eastern Europe. I did not attempt to "
        "log in from that location. I have already changed my password but am "
        "concerned my account may have been compromised.",
    ),
    "🔑  Access — SharePoint permission needed": (
        "Need access to Q4 Financial Planning SharePoint",
        "I started my new role in Finance last Monday and still do not have access "
        "to the Q4 Financial Planning SharePoint site. My manager Sarah Johnson has "
        "already approved the request in the ticketing system but I have not received "
        "access yet. The deadline for the budget submission is Friday.",
    ),
    "📧  Email — Outlook not syncing": (
        "Outlook stopped syncing new emails",
        "My Outlook desktop client stopped syncing new emails about 3 hours ago. "
        "The last email I received was at 10:15 AM. Webmail is working fine so the "
        "issue is with the desktop client. I have already tried restarting Outlook "
        "and running the Office repair tool.",
    ),
    "🖨️  Printer — 3rd floor printer offline": (
        "HP printer on 3rd floor showing offline",
        "The HP LaserJet Pro on the 3rd floor shows as offline for everyone in the "
        "department. We have tried restarting the printer and power-cycling the "
        "network switch it is connected to. The printer shows a solid green light "
        "but Windows says it is offline. Print jobs are queuing up.",
    ),
    "⚡  Performance — Workstation very slow": (
        "Workstation running extremely slow for 2 days",
        "My desktop has been running very slowly for the past two days. "
        "Applications take 3-4 minutes to open and the system feels unresponsive. "
        "CPU usage in Task Manager shows 95% even when nothing is open. "
        "I have not installed any new software recently. This is severely impacting "
        "my productivity.",
    ),
}

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────


def check_health(server_url: str) -> dict | None:
    """Poll GET /health and return the response dict, or None on failure."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def call_triage(server_url: str, text: str, backend: str) -> dict | None:
    """
    POST /triage and return the response dict, or None on error.
    Raises requests.exceptions.RequestException on network failure.
    """
    payload = {"text": text, "backend": backend}
    resp = requests.post(f"{server_url}/triage", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def badge_html(label: str, color: str, text_color: str = "white") -> str:
    """Render a colored pill badge as HTML."""
    return (
        f'<span style="'
        f"background-color: {color}; color: {text_color}; "
        f"padding: 4px 14px; border-radius: 20px; font-weight: 600; "
        f'font-size: 0.95rem;">{label}</span>'
    )


def priority_badge_html(priority: str) -> str:
    cfg = PRIORITY_CONFIG.get(priority, {"color": "#6B7280", "icon": "⚪", "border": "#D1D5DB"})
    return (
        f'<span style="'
        f"background-color: {cfg['color']}22; color: {cfg['color']}; "
        f"border: 2px solid {cfg['border']}; "
        f"padding: 4px 14px; border-radius: 20px; font-weight: 700; "
        f'font-size: 0.95rem;">{cfg["icon"]} {priority}</span>'
    )


def render_result_card(result: dict, title: str = "Result") -> None:
    """Render a single triage result as a styled card."""

    category = result.get("category", "Unknown")
    priority = result.get("priority", "Unknown")
    next_action = result.get("next_action", "")
    latency_ms = result.get("latency_ms", 0.0)
    cat_conf = result.get("cat_confidence")
    pri_conf = result.get("pri_confidence")
    reasoning = result.get("reasoning")
    cost_usd = result.get("cost_usd")
    backend_used = result.get("backend", "")

    cat_color = CATEGORY_COLORS.get(category, "#6B7280")

    st.markdown(f"#### {title}")
    st.markdown("---")

    # Category + Priority badges
    col_cat, col_pri, col_lat = st.columns([2, 2, 1])
    with col_cat:
        st.markdown("**Category**")
        st.markdown(badge_html(category, cat_color), unsafe_allow_html=True)
    with col_pri:
        st.markdown("**Priority**")
        st.markdown(priority_badge_html(priority), unsafe_allow_html=True)
    with col_lat:
        st.metric("Latency", f"{latency_ms:.0f} ms")

    st.markdown("")

    # Next action
    st.markdown("**Recommended Next Action**")
    st.info(next_action)

    # Confidence bars (finetuned only)
    if cat_conf is not None or pri_conf is not None:
        st.markdown("**Prediction Confidence**")
        conf_col1, conf_col2 = st.columns(2)
        with conf_col1:
            if cat_conf is not None:
                st.caption(f"Category confidence: {cat_conf:.1%}")
                st.progress(cat_conf)
        with conf_col2:
            if pri_conf is not None:
                st.caption(f"Priority confidence: {pri_conf:.1%}")
                st.progress(pri_conf)

    # LLM reasoning (baseline only)
    if reasoning:
        with st.expander("View LLM Reasoning", expanded=False):
            st.markdown(reasoning)

    # Cost (baseline only)
    if cost_usd is not None:
        st.caption(f"API cost: ${cost_usd:.5f} USD  |  Backend: {backend_used}")
    else:
        st.caption(f"Backend: {backend_used}")


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/ticket.png",
        width=60,
    )
    st.title("ITSM Triage Agent")
    st.caption("M8 — Live Demo")
    st.markdown("---")

    # Server URL config
    st.subheader("Server")
    server_url = st.text_input(
        "API URL",
        value=DEFAULT_SERVER_URL,
        help="URL of the running FastAPI server (M4)",
    )

    # Health check
    if st.button("Check Connection", use_container_width=True):
        health = check_health(server_url)
        if health:
            st.session_state["health"] = health
        else:
            st.session_state["health"] = None

    # Auto-check on first load
    if "health" not in st.session_state:
        st.session_state["health"] = check_health(server_url)

    health = st.session_state.get("health")
    if health:
        status = health.get("status", "unknown")
        backends = health.get("backends_available", [])
        icon = "🟢" if status == "ok" else "🟡"
        st.success(f"{icon} Server connected")
        st.caption(f"Backends available: {', '.join(backends) if backends else 'none'}")
        st.caption(f"Version: {health.get('version', '—')}")
    else:
        st.error("🔴 Cannot reach server")
        st.caption(f"Is it running at {server_url}?")
        st.code("uvicorn api.app:app --reload", language="bash")

    st.markdown("---")

    # About
    st.subheader("About")
    st.markdown(
        "**Kheiven D'Haiti**  \n"
        "B.S. Computer Science, AI Minor  \n"
        "Florida Atlantic University  \n\n"
        "[![GitHub](https://img.shields.io/badge/GitHub-Blood--Dawn-black?logo=github)]"
        "(https://github.com/Blood-Dawn/itsm-triage-agent)"
    )

    st.markdown("---")
    st.markdown(
        "**Stack:** DistilBERT + LoRA · FastAPI · Streamlit  \n"
        "**Category accuracy:** 100%  \n"
        "**Finetuned latency:** ~21 ms  \n"
        "**Baseline latency:** ~2,000 ms"
    )

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────

st.title("🎫 ITSM Ticket Triage")
st.markdown(
    "Enter a support ticket below — or pick a sample — to see the AI classify its "
    "category and priority in real time."
)

tab_single, tab_compare = st.tabs(["Single Backend", "Compare Backends"])

# ── Tab 1: Single backend ─────────────────────────────────────────────────────

with tab_single:

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Ticket Input")

        # Sample picker
        sample_key = st.selectbox(
            "Load a sample ticket",
            options=list(SAMPLE_TICKETS.keys()),
            index=0,
        )
        sample_subject, sample_body = SAMPLE_TICKETS[sample_key]

        subject = st.text_input(
            "Subject",
            value=sample_subject,
            placeholder="Brief description of the issue",
        )
        body = st.text_area(
            "Body",
            value=sample_body,
            placeholder="Detailed description of the problem...",
            height=160,
        )

        backend_choice = st.radio(
            "Backend",
            options=["finetuned", "baseline"],
            horizontal=True,
            help="finetuned: ~21ms, free | baseline: ~2000ms, uses Anthropic API",
        )

        triage_btn = st.button(
            "Triage Ticket",
            type="primary",
            use_container_width=True,
            disabled=(not subject.strip() and not body.strip()),
        )

    with col_result:
        st.subheader("Triage Result")

        if triage_btn:
            ticket_text = f"{subject.strip()}\n\n{body.strip()}".strip()
            if len(ticket_text) < 10:
                st.warning("Please enter at least a subject or body before triaging.")
            else:
                with st.spinner(f"Running {backend_choice} backend..."):
                    try:
                        result = call_triage(server_url, ticket_text, backend_choice)
                        if result and result.get("success"):
                            st.session_state["last_result"] = result
                        elif result:
                            st.error(f"Prediction failed: {result.get('error', 'unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"Cannot connect to server at {server_url}.  \n"
                            "Run: `uvicorn api.app:app --reload`"
                        )
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The baseline backend can take up to 30s.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"Server error: {e}")

        if "last_result" in st.session_state:
            render_result_card(st.session_state["last_result"], title="Prediction")
        else:
            st.markdown(
                '<div style="'
                "border: 2px dashed #374151; border-radius: 12px; "
                "padding: 40px; text-align: center; color: #9CA3AF;"
                '">Results will appear here after triaging a ticket.</div>',
                unsafe_allow_html=True,
            )

# ── Tab 2: Compare backends ───────────────────────────────────────────────────

with tab_compare:
    st.subheader("Side-by-Side Backend Comparison")
    st.caption(
        "Runs the same ticket through both backends simultaneously so you can see "
        "the accuracy/latency tradeoff in real time."
    )

    cmp_sample_key = st.selectbox(
        "Load a sample ticket",
        options=list(SAMPLE_TICKETS.keys()),
        index=0,
        key="cmp_sample",
    )
    cmp_subject, cmp_body = SAMPLE_TICKETS[cmp_sample_key]

    c1, c2 = st.columns(2)
    with c1:
        cmp_subject_in = st.text_input("Subject", value=cmp_subject, key="cmp_subj")
    with c2:
        st.markdown("")   # spacer

    cmp_body_in = st.text_area(
        "Body",
        value=cmp_body,
        height=130,
        key="cmp_body",
    )

    compare_btn = st.button(
        "Compare Both Backends",
        type="primary",
        use_container_width=True,
        disabled=(not cmp_subject_in.strip() and not cmp_body_in.strip()),
    )

    if compare_btn:
        ticket_text = f"{cmp_subject_in.strip()}\n\n{cmp_body_in.strip()}".strip()
        if len(ticket_text) < 10:
            st.warning("Please enter a subject or body first.")
        else:
            ft_result = None
            bl_result = None

            with st.spinner("Running both backends in parallel..."):
                # Run finetuned
                t0 = time.perf_counter()
                try:
                    ft_result = call_triage(server_url, ticket_text, "finetuned")
                except Exception as e:
                    st.error(f"Finetuned error: {e}")

                # Run baseline
                try:
                    bl_result = call_triage(server_url, ticket_text, "baseline")
                except requests.exceptions.Timeout:
                    st.warning("Baseline timed out — it can take up to 30s.")
                except Exception as e:
                    st.error(f"Baseline error: {e}")

                total_ms = (time.perf_counter() - t0) * 1000

            st.session_state["cmp_ft"] = ft_result
            st.session_state["cmp_bl"] = bl_result
            st.session_state["cmp_total_ms"] = total_ms

    cmp_ft = st.session_state.get("cmp_ft")
    cmp_bl = st.session_state.get("cmp_bl")
    cmp_total_ms = st.session_state.get("cmp_total_ms", 0)

    if cmp_ft or cmp_bl:
        st.markdown("---")
        if cmp_total_ms:
            st.caption(f"Total wall-clock time: {cmp_total_ms:.0f} ms (sequential)")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if cmp_ft and cmp_ft.get("success"):
                render_result_card(cmp_ft, title="Finetuned (DistilBERT + LoRA)")
            else:
                st.warning("Finetuned result unavailable.")
        with res_col2:
            if cmp_bl and cmp_bl.get("success"):
                render_result_card(cmp_bl, title="Baseline (Claude Haiku)")
            else:
                st.info(
                    "Baseline result unavailable.  \n"
                    "Set `ANTHROPIC_API_KEY` in your `.env` file to enable it."
                )

        # Agreement summary
        if cmp_ft and cmp_bl and cmp_ft.get("success") and cmp_bl.get("success"):
            st.markdown("---")
            st.subheader("Agreement Summary")
            cat_agree = cmp_ft["category"] == cmp_bl["category"]
            pri_agree = cmp_ft["priority"] == cmp_bl["priority"]
            lat_ft = cmp_ft.get("latency_ms", 0)
            lat_bl = cmp_bl.get("latency_ms", 0)
            speedup = lat_bl / lat_ft if lat_ft > 0 else 0

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric(
                    "Category agreement",
                    "Match" if cat_agree else "Differ",
                    delta=None,
                )
            with a2:
                st.metric(
                    "Priority agreement",
                    "Match" if pri_agree else "Differ",
                    delta=None,
                )
            with a3:
                st.metric(
                    "Speed advantage",
                    f"{speedup:.0f}x faster",
                    delta="finetuned wins",
                    delta_color="normal",
                )
