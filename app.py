import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import os
import requests
import io

# --- পেজ কনফিগারেশন (CapCut Pro লুক দেওয়ার জন্য) ---
st.set_page_config(page_title="AI Photo Editor Pro", layout="wide", initial_sidebar_state="expanded")

# --- ডার্ক থিম স্টাইল ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1f538d; color: white; }
    .stDownloadButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #28a745; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- এআই মডেল ডাউনলোড লজিক ---
MODEL_PATH = "EDSR_x4.pb"
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("প্রথমবার এআই মডেল ডাউনলোড হচ্ছে... দয়া করে ১ মিনিট অপেক্ষা করুন।"):
            url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
            r = requests.get(url, allow_redirects=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)

# --- মেইন ফাংশনসমূহ ---

# ১. এইচডি রেজুলেশন করার ফাংশন
def enhance_image(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("edsr", 4) # ৪ গুণ রেজুলেশন বাড়াবে
    upscaled = sr.upsample(img_cv)
    result = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

# ২. ব্যাকগ্রাউন্ড রিমুভ করার ফাংশন
def remove_bg(image):
    return remove(image)

# --- অ্যাপ ইউজার ইন্টারফেস (UI) ---

st.title("🚀 AI Photo Studio Pro")
st.subheader("পুরানো ছবিকে এইচডি করুন এবং এআই দিয়ে এডিট করুন")

# সাইডবার টুলস
st.sidebar.title("AI Tools Menu")
option = st.sidebar.selectbox("কি করতে চান সিলেক্ট করুন:", 
                             ("HD Image Enhancer", "Background Remover", "Black & White to Color"))

uploaded_file = st.sidebar.file_uploader("আপনার ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("আগের ছবি (Original)")
        st.image(original_image, use_container_width=True)
    
    # এআই প্রসেসিং বাটন
    if st.sidebar.button("এআই ম্যাজিক শুরু করুন"):
        download_model() # নিশ্চিত করা মডেলটি আছে কি না
        
        with st.spinner('AI কাজ করছে...'):
            if option == "HD Image Enhancer":
                processed_image = enhance_image(original_image)
                st.sidebar.success("এইচডি করার কাজ শেষ!")
                
            elif option == "Background Remover":
                processed_image = remove_bg(original_image)
                st.sidebar.success("ব্যাকগ্রাউন্ড রিমুভ হয়েছে!")

            elif option == "Black & White to Color":
                # এটি আরও উন্নত এআই মডেলের জন্য রাখা হয়েছে
                st.sidebar.warning("এই ফিচারটি শীঘ্রই আসছে!")
                processed_image = original_image

            with col2:
                st.success("এআই আউটপুট (Pro HD)")
                st.image(processed_image, use_container_width=True)
                
                # ডাউনলোড বাটন
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download High Res Image",
                    data=byte_im,
                    file_name="ai_pro_output.png",
                    mime="image/png"
                )
else:
    st.info("বাম পাশের সাইডবার থেকে একটি ছবি আপলোড করুন।")

# নিচের অংশ (Footer)
st.markdown("---")
st.write("Developed with AI Power | HD রেজুলেশন পাবেন ১০০%")