import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# --- ตั้งค่าหน้าจอ ---
st.set_page_config(layout="wide", page_title="Urine Analyzer")

def show_disclaimer():
    st.warning("""
      **คำเตือนและข้อควรระวัง :**
    * โปรแกรมนี้เป็นเพียงเครื่องมือช่วยตรวจวิเคราะห์ **เบื้องต้น** จากภาพถ่ายเท่านั้น ไม่ใช่การวินิจฉัยโรคโดยแพทย์
    * หากระบบตรวจพบความผิดปกติ **ควรปรึกษาแพทย์เพื่อตรวจยืนยันด้วยวิธีมาตรฐานทันที**
    """)

st.title("ระบบวิเคราะห์แถบตรวจปัสสาวะ  Urine Analyzer")
show_disclaimer()

# --- 1. ข้อมูลสีมาตรฐาน (Shifted & Calibrated) ---
COLOR_REFS = {
    "Glucose": [
        {"label": "Negative", "rgb": [115, 175, 205], "value": 0, "risk": "ปกติ"},
        {"label": "100 (+/-)", "rgb": [135, 170, 120], "value": 25, "risk": "เริ่มพบน้ำตาล"},
        {"label": "250 (+)",   "rgb": [120, 140, 90],  "value": 50, "risk": "ระดับน้ำตาลสูงปานกลาง"},
        {"label": "500 (++)",  "rgb": [100, 100, 70],  "value": 75, "risk": "ระดับน้ำตาลสูง"},
        {"label": "1000 (+++)", "rgb": [78, 62, 52],    "value": 100, "risk": "ระดับน้ำตาลสูงมาก (อันตราย)"}
    ],
    "Protein": [
        {"label": "Negative", "rgb": [188, 182, 150], "value": 0, "risk": "ปกติ"}, 
        {"label": "Trace",    "rgb": [170, 185, 120], "value": 20, "risk": "เสี่ยงโรคไตระยะต้น (ACR 30-300)"},
        {"label": "30 (+)",   "rgb": [145, 175, 120], "value": 40, "risk": "พบโปรตีน (ควรเฝ้าระวัง)"},
        {"label": "100 (++)",  "rgb": [120, 160, 120], "value": 70, "risk": "ความเสี่ยงโรคไตสูง (ACR > 300)"},
        {"label": "300 (+++)", "rgb": [95, 140, 120],  "value": 90, "risk": "ความเสี่ยงโรคไตสูงมาก"},
        {"label": "1000 (++++)", "rgb": [70, 120, 110], "value": 100, "risk": "ความเสี่ยงโรคไตอันตราย"}
    ]
}

def analyze_color(target_rgb, test_type):
    refs = COLOR_REFS[test_type]
    dists = [np.linalg.norm(np.array(target_rgb) - np.array(r["rgb"])) for r in refs]
    return refs[np.argmin(dists)]

# --- 2. ฟังก์ชันสร้าง Gauge แบบหลากสี (Colorful Gauge) ---
def create_colorful_gauge(value, label, name):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 60, 'color': "#4a4a4a"}},
        title={'text': f"<b>{name}</b><br><span style='font-size:0.8em;color:gray'>{label}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # ซ่อนแท่งสีหลักเพื่อใช้แถบสี Gradient แทน
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#c8f7c5"},   # เขียวอ่อน (ปกติ)
                {'range': [20, 50], 'color': "#fff9c4"},  # เหลือง (เริ่มเสี่ยง)
                {'range': [50, 80], 'color': "#ffecb3"},  # ส้มอ่อน (ผิดปกติ)
                {'range': [80, 100], 'color': "#ffcdd2"}  # แดงอ่อน (อันตราย)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 5},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
    return fig

# --- 3. การประมวลผลภาพ ---
def process_strip(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None, None, None, None, None
    best_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(best_cnt)
    box = cv2.boxPoints(rect).astype(int)
    
    w_std, h_std = 800, 100
    pts = box.astype("float32")
    idx = np.argsort(pts[:, 0])
    l_p, r_p = pts[idx[:2]], pts[idx[2:]]
    tl = l_p[np.argmin(l_p[:, 1])]; bl = l_p[np.argmax(l_p[:, 1])]
    tr = r_p[np.argmin(r_p[:, 1])]; br = r_p[np.argmax(r_p[:, 1])]
    
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), np.array([[0,0],[w_std-1,0],[w_std-1,h_std-1],[0,h_std-1]], dtype="float32"))
    warped = cv2.warpPerspective(image_bgr, M, (w_std, h_std))
    return image_bgr, warped, box, [tl, tr, br, bl], M

# --- 4. Main App Logic ---
uploaded_file = st.sidebar.file_uploader("Upload Test Strip", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img_bgr = cv2.cvtColor(np.array(Image.open(uploaded_file)), cv2.COLOR_RGB2BGR)
    full_img, warped, strip_box, warp_pts, M = process_strip(img_bgr)

    if warped is not None:
        st.subheader(" ผลการวิเคราะห์แยกตามพารามิเตอร์")
        col1, col2 = st.columns(2)
        
        rois = {
            "Glucose": warped[30:70, int(800*0.03):int(800*0.13)],
            "Protein": warped[30:70, int(800*0.14):int(800*0.25)]
        }

        total_risk = False
        for i, (name, roi) in enumerate(rois.items()):
            avg_rgb = cv2.mean(roi)[:3][::-1]
            res = analyze_color(avg_rgb, name)
            if res["value"] > 0: total_risk = True
            
            with [col1, col2][i]:
                # แสดง Gauge แบบมีสีสันและค่า %
                st.plotly_chart(create_colorful_gauge(res["value"], res["label"], name), use_container_width=True)
                st.markdown(f"** ผลวินิจฉัย:** {res['risk']}")
                st.image(np.full((30, 100, 3), avg_rgb, dtype=np.uint8), caption="สีจริงที่ AI ตรวจพบ")

        # สรุปผลด้านล่าง
        if total_risk:
            st.error(" ตรวจพบความเสี่ยงผิดปกติ แนะนำให้พบแพทย์เพื่อตรวจยืนยัน")
        else:
            st.success(" ผลการตรวจเบื้องต้นอยู่ในเกณฑ์ปกติ")

        # แสดงภาพยืนยันการตีกรอบ 3 สี (Combined)
        st.divider()
        st.subheader(" การตรวจจับตำแหน่ง ")
        combined_img = full_img.copy()
        cv2.drawContours(combined_img, [strip_box], 0, (0, 0, 255), 10) # แดง
        
        for color, roi_range in [((255, 255, 0), (0.03, 0.13)), ((0, 255, 255), (0.14, 0.25))]:
            pts_w = np.array([[[int(800*roi_range[0]), 30]], [[int(800*roi_range[1]), 30]], 
                               [[int(800*roi_range[1]), 70]], [[int(800*roi_range[0]), 70]]], dtype="float32")
            pts_o = cv2.perspectiveTransform(pts_w, np.linalg.inv(M)).astype(int).reshape(-1,2)
            cv2.drawContours(combined_img, [pts_o], 0, color, 5)

        st.image(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB), use_container_width=True)