import streamlit as st
import joblib
import os

# Page config 
st.set_page_config(
    page_title="Mental Health Risk Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

* { font-family: 'Inter', sans-serif; }

.left-panel {
    background: #1a2c4e;
    border-radius: 16px;
    padding: 2rem;
    color: white;
    min-height: 80vh;
}

.left-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.left-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

.right-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: white;
    font-weight: 700;
    margin-bottom: 1rem;
}

.result-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 6px solid;
}

.result-normal     { background:#f0fdf4; border-color:#22c55e; color:#166534; }
.result-depression { background:#fefce8; border-color:#eab308; color:#854d0e; }
.result-anxiety    { background:#fff7ed; border-color:#f97316; color:#9a3412; }
.result-suicidal   { background:#fef2f2; border-color:#ef4444; color:#991b1b; }

.result-label { font-size:1.3rem; font-weight:700; margin-bottom:0.3rem; }
.result-desc  { font-size:0.88rem; opacity:0.85; }

.resource-card {
    background: white;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.resource-name   { font-weight:600; color:#1a2c4e; font-size:0.88rem; }
.resource-detail { color:#64748b; font-size:0.8rem; margin-top:0.15rem; }

.section-hdr {
    font-weight: 700;
    color: white;
    font-size: 0.95rem;
    margin: 1rem 0 0.4rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 2px solid #e2e8f0;
}

.conf-bar  { background:#e2e8f0; border-radius:999px; height:7px; margin:0.2rem 0 0.6rem 0; overflow:hidden; }
.conf-fill { height:100%; border-radius:999px; }

.disclaimer {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 1.2rem;
    text-align: center;
}

.placeholder-box {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #94a3b8;
    margin-top: 2rem;
}

#MainMenu {visibility:hidden;}
footer     {visibility:hidden;}
header     {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    try:
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
        rf_model   = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
        svm_model  = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
        return vectorizer, rf_model, svm_model, True
    except:
        return None, None, None, False

vectorizer, rf_model, svm_model, models_loaded = load_models()

# Class config 
CLASS_CONFIG = {
    "Normal": {
        "emoji": "", "css": "result-normal", "color": "#22c55e",
        "desc": "No significant mental health risk indicators detected in this text.",
        "resources": {
            "Self-Care Tips": [
                ("Stay active", "Regular exercise improves mood significantly"),
                ("Sleep well", "Aim for 7–9 hours of quality sleep each night"),
                ("Stay connected", "Maintain relationships with friends and family"),
                ("Mindfulness", "Try 5 minutes of deep breathing daily"),
            ],
            "Useful Apps": [
                ("Headspace", "headspace.com — Meditation and mindfulness"),
                ("Calm", "calm.com — Sleep, meditation, relaxation"),
            ]
        }
    },
    "Depression": {
        "emoji": "", "css": "result-depression", "color": "#eab308",
        "desc": "This text shows indicators commonly associated with depression. Please seek support.",
        "resources": {
            "Crisis Helplines ": [
                ("Nigeria Suicide Prevention Initiative", "Call: 0800-7842433 — Free, 24/7"),
                ("Lagos State Mental Health", "Call: 08000-432584"),
                ("WHO Mental Health", "who.int/mental-health"),
            ],
            "Professional Help ": [
                ("Therapy in Nigeria", "therapyinnigeria.com — Find a therapist"),
                ("BetterHelp", "betterhelp.com — Online therapy"),
                ("7 Cups", "7cups.com — Free emotional support chat"),
            ],
            "Self Help ": [
                ("WHO Guides", "who.int — Free mental health guides"),
                ("YouTube", "Search: Depression self help for expert videos"),
            ]
        }
    },
    "Anxiety": {
        "emoji": "", "css": "result-anxiety", "color": "#f97316",
        "desc": "This text shows indicators commonly associated with anxiety. Support is available.",
        "resources": {
            "Immediate Relief ": [
                ("4-7-8 Breathing", "Inhale 4s → Hold 7s → Exhale 8s. Repeat 3x"),
                ("Grounding 5-4-3-2-1", "Name 5 things you see, 4 feel, 3 hear, 2 smell, 1 taste"),
                ("Progressive Muscle Relaxation", "Tense and release each muscle group from feet to head"),
            ],
            "Professional Help ": [
                ("Therapy in Nigeria", "therapyinnigeria.com — Anxiety specialists"),
                ("BetterHelp", "betterhelp.com — Online therapy sessions"),
                ("7 Cups", "7cups.com — Free emotional support"),
            ],
            "Apps 📱": [
                ("Headspace", "headspace.com — Anxiety meditations"),
                ("Calm", "calm.com — Breathing exercises and sleep tools"),
            ]
        }
    },
    "Suicidal": {
        "emoji": "", "css": "result-suicidal", "color": "#ef4444",
        "desc": "This text contains indicators of suicidal ideation. Please reach out for help immediately.",
        "resources": {
            "EMERGENCY — Call Now ": [
                ("Nigeria Emergency", "Call: 112 or 199 immediately"),
                ("Nigeria Suicide Prevention Initiative", "Call: 0800-7842433 — Free, 24/7"),
                ("Lagos State Emergency", "Call: 767 or 08000-432584"),
                ("Crisis Text Line", "Text HOME to 741741"),
            ],
            "Talk to Someone Now ": [
                ("7 Cups", "7cups.com — Free anonymous chat with trained listeners"),
                ("BetterHelp", "betterhelp.com — Connect with a licensed therapist today"),
            ],
            "Remember ": [
                ("You are not alone", "Millions of people have felt this way and found help"),
                ("This feeling will pass", "Crises are temporary — please talk to someone first"),
            ]
        }
    }
}

# Predict
def predict(text, model_choice):
    vec   = vectorizer.transform([text])
    model = rf_model if model_choice == "Random Forest" else svm_model
    pred  = model.predict(vec)[0]
    try:
        proba   = model.predict_proba(vec)[0]
        classes = model.classes_
        conf    = {c: float(p) for c, p in zip(classes, proba)}
    except:
        conf = {pred: 1.0}
    return pred, conf

#TWO PANEL LAYOUT 
left, right = st.columns([1, 1], gap="large")


# LEFT — Input Panel

with left:
    st.markdown('<div class="left-title"> Mental Health<br>Risk Detector</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="left-sub">
        A lightweight NLP-based system for detecting mental health
        risk from social media text.<br><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<b style='color:white;'> Choose Model:</b>", unsafe_allow_html=True)
    model_choice = st.radio(
        "model", ["Random Forest", "SVM"],
        label_visibility="collapsed"
    )
    acc    = "74%" if model_choice == "Random Forest" else "77%"
    detail = " n_estimators=100" if model_choice == "Random Forest" else " LinearSVC max_iter=1000"
    st.markdown(f"<small style='color:#94a3b8;'>{detail} &nbsp;|&nbsp; Accuracy: {acc}</small>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<b style='color:white;'> Enter Text:</b>", unsafe_allow_html=True)
    user_text = st.text_area(
        "text_input",
        placeholder="Type or paste a social media post here...\n\nExample:\n'I feel so hopeless and empty.\nNothing matters anymore.'",
        height=220,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button(" Analyse Text", use_container_width=True, type="primary")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<b style='color:white; font-size:0.85rem;'> Categories:</b>", unsafe_allow_html=True)
    for cls, cfg in CLASS_CONFIG.items():
        st.markdown(f"<span style='font-size:0.82rem; color:#cbd5e1;'>{cfg['emoji']} {cls}</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# RIGHT — Results Panel

with right:
    st.markdown('<div class="right-title"> Analysis Results</div>', unsafe_allow_html=True)

    if not analyse:
        st.markdown("""
        <div class="placeholder-box">
            <div style='font-size:3rem;'></div>
            <div style='font-size:1rem; font-weight:600; margin-top:1rem; color:#64748b;'>
                Enter text on the left and click<br><b>Analyse Text</b> to see results here
            </div>
            <div style='font-size:0.82rem; margin-top:0.8rem; color:#94a3b8;'>
                Results will include:<br>
                Predicted mental health status · Confidence scores · Support resources
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not user_text.strip():
        st.warning("Please enter some text in the left panel to analyse.")

    elif not models_loaded:
        st.error("Models not found. Please make sure rf_model.pkl, svm_model.pkl and vectorizer.pkl are inside a /model folder.")

    else:
        with st.spinner("Analysing..."):
            prediction, confidence = predict(user_text.strip(), model_choice)

        cfg = CLASS_CONFIG[prediction]

        # Result
        st.markdown(f"""
        <div class="result-box {cfg['css']}">
            <div class="result-label">{cfg['emoji']} {prediction}</div>
            <div class="result-desc">{cfg['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<small style='color:#64748b;'>Model: <b>{model_choice}</b> &nbsp;|&nbsp; Accuracy: <b>{acc}</b></small>", unsafe_allow_html=True)

        # Confidence scores
        if len(confidence) > 1:
            st.markdown('<div class="section-hdr"> Confidence Scores</div>', unsafe_allow_html=True)
            color_map = {"Normal":"#22c55e","Depression":"#eab308","Anxiety":"#f97316","Suicidal":"#ef4444"}
            for cls in ["Normal", "Depression", "Anxiety", "Suicidal"]:
                score = confidence.get(cls, 0.0)
                pct   = round(score * 100, 1)
                col   = color_map.get(cls, "#94a3b8")
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-top:0.4rem;'>
                    <span>{CLASS_CONFIG[cls]['emoji']} {cls}</span>
                    <span style='font-weight:600;'>{pct}%</span>
                </div>
                <div class='conf-bar'>
                    <div class='conf-fill' style='width:{pct}%;background:{col};'></div>
                </div>
                """, unsafe_allow_html=True)

        # Resources
        st.markdown('<div class="section-hdr"> Support & Resources</div>', unsafe_allow_html=True)
        for section, items in cfg["resources"].items():
            st.markdown(f"**{section}**")
            for name, detail in items:
                st.markdown(f"""
                <div class='resource-card'>
                    <div class='resource-name'>{name}</div>
                    <div class='resource-detail'>{detail}</div>
                </div>
                """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='disclaimer'>
          <b>Disclaimer:</b> This tool is for research and educational purposes only.
        It is NOT a clinical diagnostic tool and should NOT replace professional medical advice.
        If you or someone you know is in crisis, please contact a mental health professional immediately.
        </div>
        """, unsafe_allow_html=True)

