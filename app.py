import streamlit as st
import joblib
import numpy as np
import random

# Page Configuration
st.set_page_config(page_title="MindCare AI", page_icon="🧠", layout="centered")

# --- CUSTOM CSS: Clean & Professional UI ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }
    .main-header { text-align: center; color: #2c3e50; font-size: 42px; font-weight: 700; margin-bottom: 5px; }
    .sub-text { text-align: center; color: #4a90e2; font-size: 18px; margin-bottom: 30px; }
    
    .result-card {
        background: white; padding: 25px; border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 10px solid #3b82f6;
        margin-top: 20px; color: #2d3748;
    }
    .pill {
        display: inline-block; background: #e3f2fd; color: #1976d2;
        padding: 8px 15px; border-radius: 50px; font-size: 14px;
        margin: 5px; font-weight: 600; border: 1px solid #bbdefb;
    }
    .metric-box {
        background: #f8fafc; padding: 15px; border-radius: 12px;
        display: flex; justify-content: space-around; margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def load_resources():
    try:
        m = joblib.load('mental_health_model.pkl')
        t = joblib.load('tfidf_vectorizer.pkl')
        return m, t
    except: return None, None

model, tfidf = load_resources()

# --- Header ---
st.markdown("<h1 class='main-header'>MindCare AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Intelligent Mental Health Detection & Support</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Error: Model files not found.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("How are you feeling? Speak your heart out..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. Prediction Logic (ML Model)
        user_input = prompt.lower()
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        confidence = np.max(model.predict_proba(vec)[0])

        # --- 2. UPDATED ADVANCED HYBRID LOGIC ---
        
        # Keywords Definitions
        harm_words = ['hurt', 'kill', 'suicide', 'die', 'death', 'end my life', 'harm']
        stress_words = ['tension', 'pressure', 'handle', 'burden', 'exhausted', 'office', 'workload', 'overwhelmed']
        dep_words = ['sad', 'cry', 'lonely', 'hopeless', 'depressed', 'unhappy']
        negations = ['not ', 'no ', "don't ", "never ", "n't"]
        
        # New Keywords for missing classes
        bipolar_keywords = ['mood swings', 'high and low', 'extreme energy', 'uncontrollable mood', 'top of the world']
        personality_keywords = ['trust', 'unstable relationship', 'paranoid', 'fear of abandonment', 'trusting anyone']

        # Logic A: Safety Override (Highest Priority)
        if any(word in user_input for word in harm_words):
            prediction = "Suicidal"
            confidence = 1.0
        
        # Logic B: Bipolar Detection
        elif any(word in user_input for word in bipolar_keywords):
            prediction = "Bipolar"
            confidence = 0.95

        # Logic C: Personality Disorder Detection
        elif any(word in user_input for word in personality_keywords):
            prediction = "Personality Disorder"
            confidence = 0.94

        # Logic D: Stress/Tension
        elif any(word in user_input for word in stress_words):
            if prediction == "Normal" or confidence < 0.80:
                prediction = "Stress"
                confidence = 0.90
        
        # Logic E: Depression handling with negation check
        elif any(word in user_input for word in dep_words) and not any(neg in user_input for neg in negations):
            if prediction == "Normal":
                prediction = "Depression"
                confidence = 0.82

        # Logic F: Final Negation Check (Turns results to Normal)
        elif any(neg in user_input for neg in negations):
            prediction = "Normal"
            confidence = 0.95

        # --- 3. DISPLAY RESULTS ---
        # Color coding for different classes
        colors = {
            "Suicidal": "#e53e3e",
            "Normal": "#38a169",
            "Depression": "#d97706",
            "Anxiety": "#d97706",
            "Stress": "#d97706",
            "Bipolar": "#805ad5",
            "Personality Disorder": "#3182ce"
        }
        accent_color = colors.get(prediction, "#3b82f6")
        
        recs = {
            "Normal": ["Maintain Routine", "Gratitude Practice", "Stay Active"],
            "Depression": ["Talk to a Friend", "Light Movement", "Consult a Specialist"],
            "Anxiety": ["4-7-8 Breathing", "Limit Caffeine", "Grounding Exercise"],
            "Stress": ["Take a Break", "Organize Work", "Digital Detox"],
            "Suicidal": ["🚨 Call Crisis Helpline", "🚨 Visit Emergency Room", "🚨 Alert Family"],
            "Bipolar": ["Consistent Sleep", "Mood Tracking", "Professional Review"],
            "Personality Disorder": ["Therapy Skills", "Self-Reflection", "Support Group"]
        }

        pills = "".join([f"<span class='pill'>{r}</span>" for r in recs.get(prediction, ["Stay Strong"])])
        
        ai_reply = f"""
        <div class='result-card' style='border-left-color: {accent_color};'>
            <p style='font-size: 1.1em; font-weight: 500;'>AI Analysis Result:</p>
            <div class='metric-box'>
                <div style='text-align: center;'>
                    <p style='color: #718096; font-size: 0.8em; margin:0;'>PREDICTION</p>
                    <p style='color: {accent_color}; font-size: 24px; font-weight: 800; margin:0;'>{prediction}</p>
                </div>
                <div style='text-align: center;'>
                    <p style='color: #718096; font-size: 0.8em; margin:0;'>CONFIDENCE</p>
                    <p style='color: #2d3748; font-size: 24px; font-weight: 800; margin:0;'>{confidence:.2f}</p>
                </div>
            </div>
            <p style='font-weight: bold;'>💡 Recommendations:</p>
            <div>{pills}</div>
        </div>
        """

        with st.chat_message("assistant"):
            st.markdown(ai_reply, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})