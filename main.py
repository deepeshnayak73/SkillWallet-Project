"""Multimodal Construction Intelligence Platform."""

from __future__ import annotations

import json
import math
from html import escape
from io import BytesIO
from typing import Any, Callable
from urllib.parse import quote

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError


APP_TITLE = "Multimodal Construction Intelligence Platform"
MODEL_NAME = "gemini-2.5-flash"
SEVERITIES = {"Low", "Medium", "High", "Critical"}
DEFECT_TYPES = {
    "Hairline",
    "Settlement",
    "Structural Shear",
    "Thermal Expansion",
    "Spalling",
}
URGENCIES = {"Low", "Medium", "High", "Critical"}
PPE_ITEMS = {
    "helmet": "Hard hat / helmet",
    "high_visibility_vest": "High-visibility vest",
    "steel_toe_boots": "Steel-toe boots",
    "safety_harness": "Fall-protection harness",
}
STRUCTURES = {
    "Residential": {"cost": 1800, "steel": 4.0, "cement": 0.40},
    "Commercial": {"cost": 2600, "steel": 4.5, "cement": 0.45},
    "Industrial": {"cost": 1500, "steel": 3.5, "cement": 0.35},
    "Institutional": {"cost": 2200, "steel": 4.2, "cement": 0.42},
}
CITY_TIERS = {
    "Tier 1": {"cost": 1.25, "market": 1.60},
    "Tier 2": {"cost": 1.00, "market": 1.30},
    "Tier 3": {"cost": 0.85, "market": 1.00},
}


def _read_secret_key() -> str:
    """Read an optional key from Streamlit secrets without requiring a secrets file."""
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except (FileNotFoundError, RuntimeError, st.errors.StreamlitSecretNotFoundError):
        return ""


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, including a fenced JSON block if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")
    return parsed


def _validate_safety_report(report: dict[str, Any]) -> dict[str, Any]:
    score = report.get("safety_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 100
    ):
        raise ValueError("Safety report has an invalid safety_score (expected 0–100).")
    if not isinstance(report.get("overall_status"), str):
        raise ValueError("Safety report is missing a valid overall_status.")
    ppe = report.get("ppe_detection")
    if not isinstance(ppe, dict):
        raise ValueError("Safety report is missing the ppe_detection checklist.")
    for key in PPE_ITEMS:
        item = ppe.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("detected"), bool):
            raise ValueError(f"Safety report has an invalid PPE entry: {key}.")
        if not isinstance(item.get("notes", ""), str):
            raise ValueError(f"Safety report has invalid PPE notes: {key}.")
    hazards = report.get("hazards")
    if not isinstance(hazards, list):
        raise ValueError("Safety report is missing the hazards list.")
    for hazard in hazards:
        if (
            not isinstance(hazard, dict)
            or not isinstance(hazard.get("hazard"), str)
            or hazard.get("severity") not in SEVERITIES
            or not isinstance(hazard.get("remedy"), str)
        ):
            raise ValueError("Safety report contains an invalid hazard entry.")
    return report


def _validate_crack_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("defect_type") not in DEFECT_TYPES:
        raise ValueError("Crack report has an unrecognized defect_type.")
    width = report.get("estimated_crack_width_mm")
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or width < 0
        or (isinstance(width, float) and not math.isfinite(width))
    ):
        raise ValueError("Crack report has an invalid estimated crack width.")
    for field in ("root_causes", "recommended_repairs"):
        if not isinstance(report.get(field), list) or not all(
            isinstance(item, str) for item in report[field]
        ):
            raise ValueError(f"Crack report has an invalid {field} list.")
    if report.get("urgency") not in URGENCIES:
        raise ValueError("Crack report has an invalid urgency.")
    if not isinstance(report.get("observation", ""), str):
        raise ValueError("Crack report has an invalid observation.")
    return report


def _analyze_image(
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable JPG or PNG image.") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    response_text = response.text
    if not response_text:
        raise ValueError("Gemini returned an empty response. Please try again.")
    return validator(_parse_json_response(response_text))


def _demo_safety_report() -> dict[str, Any]:
    return {
        "safety_score": 72,
        "overall_status": "Needs attention",
        "ppe_detection": {
            "helmet": {"detected": True, "notes": "Illustrative demo result."},
            "high_visibility_vest": {"detected": True, "notes": "Illustrative demo result."},
            "steel_toe_boots": {"detected": False, "notes": "Not confirmed in demo mode."},
            "safety_harness": {"detected": False, "notes": "Not confirmed in demo mode."},
        },
        "hazards": [
            {
                "hazard": "Unprotected elevated work area",
                "severity": "High",
                "remedy": "Provide compliant edge protection and verify fall-protection controls.",
            },
            {
                "hazard": "Material storage near a walkway",
                "severity": "Medium",
                "remedy": "Clear and mark the access route; secure stored materials.",
            },
        ],
    }


def _demo_crack_report() -> dict[str, Any]:
    return {
        "defect_type": "Hairline",
        "estimated_crack_width_mm": 0.3,
        "root_causes": ["Illustrative demo result; no image analysis was performed."],
        "recommended_repairs": [
            "Have a qualified engineer assess the crack and monitor for progression."
        ],
        "urgency": "Low",
        "observation": "Demonstration output only. Upload an image and configure Gemini for an assessment.",
    }


def _render_status(status: str) -> None:
    normalized = status.strip().lower()
    color = (
        "#b42318"
        if any(word in normalized for word in ("critical", "unsafe", "high"))
        else "#9a6700"
        if any(word in normalized for word in ("attention", "medium", "moderate"))
        else "#067647"
    )
    st.markdown(
        f'<span style="display:inline-block;background:{color};color:white;'
        f'padding:.35rem .75rem;border-radius:999px;font-weight:700">{escape(status)}</span>',
        unsafe_allow_html=True,
    )


def _render_safety_report(report: dict[str, Any], demo: bool) -> None:
    st.subheader("Site safety report")
    if demo:
        st.info("Demonstration mode — these are sample results, not an analysis of your image.")
    left, right = st.columns(2)
    left.metric("Safety score", f"{report['safety_score']}/100")
    with right:
        st.write("Overall status")
        _render_status(report["overall_status"])
    st.markdown("#### PPE checklist")
    cols = st.columns(2)
    for index, (key, label) in enumerate(PPE_ITEMS.items()):
        item = report["ppe_detection"][key]
        with cols[index % 2]:
            st.write(f"{'✅' if item['detected'] else '❌'} **{label}**")
            if item.get("notes"):
                st.caption(item["notes"])
    st.markdown("#### Identified hazards")
    if report["hazards"]:
        st.dataframe(report["hazards"], use_container_width=True, hide_index=True)
    else:
        st.success("No hazards identified in the provided image.")
    st.download_button(
        "Download safety report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="site-safety-report.json",
        mime="application/json",
        key="download_safety_report",
    )


def _render_crack_report(report: dict[str, Any], demo: bool) -> None:
    st.subheader("Crack diagnostics")
    if demo:
        st.info("Demonstration mode — these are sample results, not an analysis of your image.")
    columns = st.columns(3)
    columns[0].metric("Defect type", report["defect_type"])
    columns[1].metric("Estimated width", f"{report['estimated_crack_width_mm']} mm")
    with columns[2]:
        st.write("Urgency")
        _render_status(report["urgency"])
    st.markdown("#### Possible root causes")
    for cause in report["root_causes"]:
        st.write(f"- {cause}")
    st.markdown("#### Recommended engineering actions")
    for repair in report["recommended_repairs"]:
        st.write(f"- {repair}")
    if report.get("observation"):
        st.info(report["observation"])
    st.download_button(
        "Download diagnostics (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="crack-diagnostics.json",
        mime="application/json",
        key="download_crack_report",
    )


def _format_inr_lakhs(amount: float) -> str:
    lakhs = amount / 100_000
    return f"₹{lakhs:,.2f} Lakhs" if lakhs < 100 else f"₹{lakhs / 100:,.2f} Crores"


def _planning_tab() -> None:
    st.subheader("Planning & real estate valuation")
    with st.form("planning_inputs"):
        col1, col2 = st.columns(2)
        structure = col1.selectbox("Structure type", list(STRUCTURES))
        area = col2.number_input("Built-up area (sq ft)", min_value=100, max_value=1_000_000, value=2_000, step=100)
        tier = col1.selectbox("Location tier", list(CITY_TIERS))
        completion = col2.slider("Completion", min_value=0, max_value=100, value=35, format="%d%%")
        submitted = st.form_submit_button("Update estimate", type="primary")

    if submitted or "planning_estimate" not in st.session_state:
        rates = STRUCTURES[structure]
        location = CITY_TIERS[tier]
        budget = area * rates["cost"] * location["cost"]
        incurred = budget * completion / 100
        valuation = budget * location["market"] * completion / 100
        steel_kg = area * rates["steel"]
        cement_bags = area * rates["cement"]
        st.session_state["planning_estimate"] = {
            "structure": structure,
            "area": area,
            "tier": tier,
            "completion": completion,
            "budget": budget,
            "incurred": incurred,
            "valuation": valuation,
            "steel_kg": steel_kg,
            "cement_bags": cement_bags,
        }
    estimate = st.session_state["planning_estimate"]
    st.caption(
        "Indicative planning estimates only. Costs vary by specification, local rates, approvals, "
        "and site conditions. Market valuation uses a simplified tier-based multiplier."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Estimated project budget", _format_inr_lakhs(estimate["budget"]))
    metric_columns[1].metric("Capital expenditure incurred", _format_inr_lakhs(estimate["incurred"]))
    metric_columns[2].metric("Indicative current asset value", _format_inr_lakhs(estimate["valuation"]))
    st.markdown("#### Material planning")
    material_columns = st.columns(2)
    material_columns[0].metric(
        "Steel requirement", f"{estimate['steel_kg']:,.0f} kg",
        f"{estimate['steel_kg'] * estimate['completion'] / 100:,.0f} kg at current progress",
    )
    material_columns[1].metric(
        "Cement requirement", f"{estimate['cement_bags']:,.0f} bags",
        f"{estimate['cement_bags'] * estimate['completion'] / 100:,.0f} bags at current progress",
    )
    st.markdown("#### Project milestone progress")
    st.progress(estimate["completion"], text=f"{estimate['completion']}% complete")
    milestones = [
        ("Planning & approvals", 10),
        ("Foundation", 25),
        ("Structure", 60),
        ("Finishing", 90),
        ("Handover", 100),
    ]
    st.write(
        "  ·  ".join(
            f"{'✅' if estimate['completion'] >= target else '○'} {name}"
            for name, target in milestones
        )
    )
    st.markdown("#### Conceptual 3D rendering prompt")
    rendering_prompt = (
        f"Photorealistic conceptual 3D architectural visualization of a {estimate['structure'].lower()} "
        f"building in a {estimate['tier']} Indian city, about {estimate['area']:,} square feet, "
        "contemporary climate-responsive design, realistic materials, daylight, architectural rendering"
    )
    st.text_area("Prompt for Pollinations AI", value=rendering_prompt, height=95, disabled=True)
    image_url = f"https://image.pollinations.ai/prompt/{quote(rendering_prompt, safe='')}"
    st.link_button("Open conceptual rendering with Pollinations AI", image_url)


WEATHER_ADVISORIES = {
    "High Wind": (
        "Pause crane lifts, work at height, and suspended-load operations when wind exceeds "
        "equipment and site limits. Secure loose materials and inspect temporary structures."
    ),
    "Heavy Rain": (
        "Protect excavations and electrical equipment from water ingress. Isolate flooded circuits, "
        "provide safe drainage, and inspect scaffolds and access routes before resuming work."
    ),
    "Extreme Heat": (
        "Schedule strenuous work for cooler hours, provide shade and drinking water, and use work/rest "
        "cycles. Protect fresh concrete and follow the approved curing specification."
    ),
    "Normal": (
        "Continue routine pre-task briefings, PPE checks, housekeeping, and equipment inspections. "
        "Monitor local forecasts and reassess conditions throughout the shift."
    ),
}


def _assistant_tab(api_key: str) -> None:
    st.subheader("Daily site assistant")
    weather = st.selectbox("Current weather condition", list(WEATHER_ADVISORIES))
    st.warning(f"**Weather-aware safety advisory — {weather}:** {WEATHER_ADVISORIES[weather]}")
    st.caption("General guidance only. Follow your site's approved safety plan and a qualified engineer's direction.")

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []
    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a construction planning or safety question")
    if question:
        st.session_state.assistant_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        if not api_key:
            answer = (
                f"**Demonstration assistant:** For {weather.lower()} conditions, use this site reminder: "
                f"{WEATHER_ADVISORIES[weather]} This sample response is not tailored to your project. "
                "Add a Gemini key in the sidebar for conversational AI guidance. For urgent site safety "
                "or structural concerns, contact the responsible professional."
            )
        else:
            try:
                client = genai.Client(api_key=api_key)
                history = [
                    {
                        "role": "model" if item["role"] == "assistant" else "user",
                        "parts": [item["content"]],
                    }
                    for item in st.session_state.assistant_messages
                ]
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a cautious construction site assistant. Give practical, concise "
                            "general guidance, state uncertainty, and defer structural design or "
                            "site-specific safety decisions to qualified professionals."
                        )
                    ),
                )
                answer = response.text
                if not answer:
                    raise ValueError("Gemini returned an empty response. Please try again.")
            except Exception as exc:
                st.error(f"Could not get a Gemini response: {exc}")
                st.session_state.assistant_messages.pop()
                return
        st.session_state.assistant_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🏗️", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {background:#f5f8fc;padding:1rem;border-radius:12px;border:1px solid #e5eaf1;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🏗️ Multimodal Construction Intelligence Platform")
    st.caption("AI-assisted site insights, planning estimates, and weather-aware guidance")

    with st.sidebar:
        st.header("Platform controls")
        api_key = st.text_input(
            "Google Gemini API key",
            value=_read_secret_key(),
            type="password",
            help="Optional. Used only for Gemini requests in this session. Leave blank for demonstration mode.",
        ).strip()
        st.caption("Without a key, image reports use clearly labeled sample data and chat is unavailable.")
        st.divider()
        st.markdown(
            "**Architecture**\n\n"
            "Streamlit UI → Gemini multimodal analysis (optional) → validated JSON reports. "
            "Planning calculations run locally; no external database is used."
        )
        st.markdown(
            "**Features**\n\n"
            "1. Site Safety & PPE Audit\n"
            "2. Crack Diagnostics\n"
            "3. Planning & Valuation\n"
            "4. Daily Site Assistant"
        )
        st.caption("Visual AI findings are preliminary and are not a compliance certification or engineering inspection.")

    safety_tab, crack_tab, planning_tab, assistant_tab = st.tabs(
        [
            "🦺 Site Safety & PPE Audit",
            "🧱 Crack Diagnostics",
            "📊 Planning & Valuation",
            "🌦️ Daily Site Assistant",
        ]
    )
    with safety_tab:
        st.subheader("Site safety & PPE audit")
        st.write("Upload a clear site photo for a preliminary visual audit.")
        image_file = st.file_uploader(
            "Construction site photo (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="safety_image",
        )
        if st.button("Analyze site safety", type="primary", disabled=image_file is None):
            if not api_key:
                report, demo = _demo_safety_report(), True
            else:
                prompt = (
                    "Inspect this construction-site image for visible worker PPE and site hazards. "
                    "Provide preliminary visual observations only; do not claim certification or infer "
                    "details that are not visible. Compare PPE observations with applicable OSHA "
                    "guidance. Mention IS 456 only for relevant concrete context; it is not a PPE "
                    "standard. "
                    "Return only a JSON object with: safety_score (number 0-100), overall_status (string), "
                    "ppe_detection (object with helmet, high_visibility_vest, steel_toe_boots, "
                    "safety_harness; each has detected (boolean) and notes (string)), and hazards "
                    "(array of objects with hazard, severity [Low, Medium, High, Critical], remedy). "
                    "Treat not-visible PPE as not confirmed, not necessarily absent."
                )
                try:
                    report = _analyze_image(api_key, image_file.getvalue(), prompt, _validate_safety_report)
                    demo = False
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"Could not create a valid safety report: {exc}")
                    report = None
                except Exception as exc:
                    st.error(f"Gemini safety analysis failed: {exc}")
                    report = None
            if report is not None:
                st.session_state["safety_report"] = report
                st.session_state["safety_report_demo"] = demo
        if "safety_report" in st.session_state:
            _render_safety_report(
                st.session_state["safety_report"],
                st.session_state["safety_report_demo"],
            )

    with crack_tab:
        st.subheader("Crack diagnostics & structural integrity")
        st.write("Upload a focused, well-lit concrete or wall surface photo.")
        crack_image = st.file_uploader(
            "Concrete or wall photo (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="crack_image",
        )
        if st.button("Analyze surface", type="primary", disabled=crack_image is None):
            if not api_key:
                report, demo = _demo_crack_report(), True
            else:
                prompt = (
                    "Visually inspect this concrete or wall surface for a possible defect. Return only "
                    "a JSON object with defect_type (one of Hairline, Settlement, Structural Shear, "
                    "Thermal Expansion, Spalling), estimated_crack_width_mm (number), root_causes "
                    "(array of strings), recommended_repairs (array of strings, consider epoxy injection "
                    "or structural re-grouting only when appropriate), urgency (Low, Medium, High, or "
                    "Critical), and observation (string). State uncertainty when scale or context is "
                    "missing. Do not provide a structural certification; recommend on-site assessment "
                    "for potentially structural damage."
                )
                try:
                    report = _analyze_image(api_key, crack_image.getvalue(), prompt, _validate_crack_report)
                    demo = False
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"Could not create valid crack diagnostics: {exc}")
                    report = None
                except Exception as exc:
                    st.error(f"Gemini crack analysis failed: {exc}")
                    report = None
            if report is not None:
                st.session_state["crack_report"] = report
                st.session_state["crack_report_demo"] = demo
        if "crack_report" in st.session_state:
            _render_crack_report(
                st.session_state["crack_report"],
                st.session_state["crack_report_demo"],
            )

    with planning_tab:
        _planning_tab()
    with assistant_tab:
        _assistant_tab(api_key)

    st.divider()
    st.caption(
        "For planning and educational support only. Follow applicable codes, approved project documents, "
        "site procedures, and advice from licensed/qualified professionals."
    )


if __name__ == "__main__":
    main()
