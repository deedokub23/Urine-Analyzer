import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Urine Strip Analyzer", layout="centered")

st.title("🧪 Urine Strip Analyzer (HSV + Risk %)")

# ------------------ FUNCTIONS ------------------

def image_to_hsv(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img

def get_grid_colors(hsv_img, rows=2, cols=5):
    h, w, _ = hsv_img.shape
    grid_colors = []

    for i in range(rows):
        for j in range(cols):
            y1 = int(i * h / rows)
            y2 = int((i + 1) * h / rows)
            x1 = int(j * w / cols)
            x2 = int((j + 1) * w / cols)

            cell = hsv_img[y1:y2, x1:x2]
            avg_color = np.mean(cell.reshape(-1, 3), axis=0)
            grid_colors.append(avg_color)

    return grid_colors

def color_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

def calculate_similarity(ref_colors, test_colors):
    similarities = []
    for r, t in zip(ref_colors, test_colors):
        dist = color_distance(r, t)
        sim = max(0, 100 - dist)  # แปลงเป็น %
        similarities.append(sim)
    return similarities

def draw_grid(image, rows=2, cols=5):
    img = np.array(image).copy()
    h, w, _ = img.shape

    for i in range(1, rows):
        cv2.line(img, (0, int(i*h/rows)), (w, int(i*h/rows)), (0,255,0), 2)

    for j in range(1, cols):
        cv2.line(img, (int(j*w/cols), 0), (int(j*w/cols), h), (0,255,0), 2)

    return img

# ------------------ UI ------------------

ref_file = st.file_uploader("📦 อัปโหลดภาพ 'ข้างกล่อง (Reference)'", type=["jpg","png"])
test_file = st.file_uploader("🧪 อัปโหลดภาพ 'แถบตรวจจริง'", type=["jpg","png"])

if ref_file and test_file:
    ref_img = Image.open(ref_file).resize((400,200))
    test_img = Image.open(test_file).resize((400,200))

    st.subheader("📷 ภาพพร้อมกรอบสแกน")
    col1, col2 = st.columns(2)

    with col1:
        st.image(draw_grid(ref_img), caption="Reference")

    with col2:
        st.image(draw_grid(test_img), caption="Test")

    # แปลง HSV
    ref_hsv = image_to_hsv(ref_img)
    test_hsv = image_to_hsv(test_img)

    # ดึงสีแต่ละช่อง
    ref_colors = get_grid_colors(ref_hsv)
    test_colors = get_grid_colors(test_hsv)

    # เปรียบเทียบ
    similarities = calculate_similarity(ref_colors, test_colors)

    st.subheader("📊 ผลลัพธ์")

    labels = [f"ช่อง {i+1}" for i in range(len(similarities))]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=similarities))

    fig.update_layout(
        title="เปอร์เซ็นต์ความใกล้เคียงสี (ยิ่งต่ำ = เสี่ยง)",
        yaxis_title="%",
        xaxis_title="ช่องตรวจ"
    )

    st.plotly_chart(fig)

    # แปลเป็นความเสี่ยง
    st.subheader("⚠️ วิเคราะห์ความเสี่ยง")
    for i, sim in enumerate(similarities):
        risk = 100 - sim

        if risk < 20:
            level = "🟢 ปกติ"
        elif risk < 50:
            level = "🟡 เฝ้าระวัง"
        else:
            level = "🔴 เสี่ยงสูง"

        st.write(f"ช่อง {i+1}: {level} ({risk:.1f}%)")