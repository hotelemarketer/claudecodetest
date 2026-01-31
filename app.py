import streamlit as st
import numpy as np
from PIL import Image
import face_recognition
import io

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Spitting Image", page_icon="👶", layout="centered")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background-color: #FFFFFF;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title */
.hero-title {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 3.4rem;
    color: #2C2C2C;
    margin-top: 1.5rem;
    margin-bottom: 0.1rem;
    letter-spacing: -1px;
}
.hero-sub {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 300;
    font-size: 1.15rem;
    color: #888;
    margin-bottom: 2.5rem;
}

/* Upload cards */
.upload-label {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.3rem;
    color: #444;
}
.upload-label-dad { color: #6FA3B8; }
.upload-label-mom { color: #D4847A; }
.upload-label-child { color: #888; }

/* Photo frames */
.photo-frame {
    border-radius: 18px;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto 1rem auto;
    max-width: 220px;
}
.photo-frame img {
    width: 100%;
    border-radius: 18px;
}
.frame-dad { border: 4px solid #AEC6CF; }
.frame-mom { border: 4px solid #FFB7B2; }
.frame-child { border: 4px solid #D5D5D5; }

/* Analyze button */
div.stButton > button {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    background: linear-gradient(135deg, #AEC6CF 0%, #FFB7B2 100%);
    color: #fff;
    border: none;
    border-radius: 40px;
    padding: 0.7rem 2.8rem;
    display: block;
    margin: 2rem auto;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    color: #fff;
    border: none;
}
div.stButton > button:active, div.stButton > button:focus {
    color: #fff;
    border: none;
    background: linear-gradient(135deg, #AEC6CF 0%, #FFB7B2 100%);
}

/* Verdict */
.verdict {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #2C2C2C;
    margin: 1.5rem 0 0.5rem 0;
}

/* Progress bar */
.pbar-container {
    width: 80%;
    max-width: 520px;
    margin: 0.8rem auto 0.3rem auto;
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    height: 36px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.pbar-dad {
    background: #AEC6CF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #fff;
}
.pbar-mom {
    background: #FFB7B2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #fff;
}
.pbar-labels {
    width: 80%;
    max-width: 520px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
    font-size: 0.82rem;
    color: #999;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Spitting Image</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload photos to discover who the child resembles more</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_rgb_array(uploaded_file) -> np.ndarray:
    """Read an uploaded file into an RGB numpy array without touching disk."""
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    img = img.convert("RGB")
    return np.array(img)


def encode_face(image_array: np.ndarray, label: str):
    """Return face encoding or set an error via st.error and return None."""
    locations = face_recognition.face_locations(image_array, model="hog")
    if len(locations) == 0:
        st.error(f"😕 No face detected in the **{label}** photo. Please upload a clear photo with one visible face.")
        return None
    if len(locations) > 1:
        st.error(f"😕 Multiple faces detected in the **{label}** photo. Please upload a photo with only one person.")
        return None
    encodings = face_recognition.face_encodings(image_array, known_face_locations=locations)
    return encodings[0]


def image_to_data_uri(uploaded_file) -> str:
    """Return a base64 data URI for inline display."""
    import base64
    data = base64.b64encode(uploaded_file.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"

# ---------------------------------------------------------------------------
# Upload UI
# ---------------------------------------------------------------------------
col_dad, col_child, col_mom = st.columns([1, 1, 1], gap="large")

with col_dad:
    st.markdown('<div class="upload-label upload-label-dad">Father\'s Photo</div>', unsafe_allow_html=True)
    dad_file = st.file_uploader("Upload father", type=["jpg", "jpeg", "png"], key="dad", label_visibility="collapsed")

with col_child:
    st.markdown('<div class="upload-label upload-label-child">Child\'s Photo</div>', unsafe_allow_html=True)
    child_file = st.file_uploader("Upload child", type=["jpg", "jpeg", "png"], key="child", label_visibility="collapsed")

with col_mom:
    st.markdown('<div class="upload-label upload-label-mom">Mother\'s Photo</div>', unsafe_allow_html=True)
    mom_file = st.file_uploader("Upload mother", type=["jpg", "jpeg", "png"], key="mom", label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Analyze button
# ---------------------------------------------------------------------------
analyze = st.button("Analyze Lineage")

if analyze:
    if not (dad_file and mom_file and child_file):
        st.warning("Please upload all three photos before analyzing.")
        st.stop()

    with st.spinner("Analyzing facial features…"):
        dad_img = pil_to_rgb_array(dad_file)
        mom_img = pil_to_rgb_array(mom_file)
        child_img = pil_to_rgb_array(child_file)

        dad_enc = encode_face(dad_img, "Father")
        mom_enc = encode_face(mom_img, "Mother")
        child_enc = encode_face(child_img, "Child")

        if dad_enc is None or mom_enc is None or child_enc is None:
            st.stop()

        # --- Math ---
        dist_dad = face_recognition.face_distance([dad_enc], child_enc)[0]
        dist_mom = face_recognition.face_distance([mom_enc], child_enc)[0]

        # Lower distance = closer match → invert
        inv_dad = 1.0 / (dist_dad + 1e-9)
        inv_mom = 1.0 / (dist_mom + 1e-9)
        total_inv = inv_dad + inv_mom
        pct_dad = round(inv_dad / total_inv * 100, 1)
        pct_mom = round(inv_mom / total_inv * 100, 1)

        winner = "Father" if pct_dad >= pct_mom else "Mother"
        accent = "#AEC6CF" if winner == "Father" else "#FFB7B2"

    # --- The Reveal ---
    st.markdown("---")

    r_dad, r_child, r_mom = st.columns([1, 1, 1], gap="large")
    with r_dad:
        st.markdown(f'<div class="photo-frame frame-dad"><img src="{image_to_data_uri(dad_file)}"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;font-weight:600;color:#6FA3B8;">{pct_dad}%</div>', unsafe_allow_html=True)
    with r_child:
        st.markdown(f'<div class="photo-frame frame-child"><img src="{image_to_data_uri(child_file)}"></div>', unsafe_allow_html=True)
    with r_mom:
        st.markdown(f'<div class="photo-frame frame-mom"><img src="{image_to_data_uri(mom_file)}"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;font-weight:600;color:#D4847A;">{pct_mom}%</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="verdict">The child is the <span style="color:{accent}">Spitting Image</span> of the {winner}!</div>', unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
    <div class="pbar-container">
        <div class="pbar-dad" style="width:{pct_dad}%;">{pct_dad}%</div>
        <div class="pbar-mom" style="width:{pct_mom}%;">{pct_mom}%</div>
    </div>
    <div class="pbar-labels">
        <span>👨 Father</span>
        <span>👩 Mother</span>
    </div>
    """, unsafe_allow_html=True)
