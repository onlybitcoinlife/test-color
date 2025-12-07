import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import time

# Page Config
st.set_page_config(page_title="AI Beauty Advisor", page_icon="💄")

# --- 🎨 CSS for Live Face Guide ---
st.markdown(
    """
    <style>
    /* Camera Input Container */
    div[data-testid="stCameraInput"] {
        position: relative;
    }
    
    /* Face Guide Overlay */
    div[data-testid="stCameraInput"]::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 250px; /* Adjust width for face */
        height: 330px; /* Adjust height for face */
        border: 3px dashed rgba(255, 255, 255, 0.7); /* Dotted white line */
        border-radius: 50% 50% 50% 50% / 40% 40% 60% 60%; /* Inverted Egg shape (Wider top, narrower bottom) */
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); /* Dim the outside */
        pointer-events: none; /* Allow clicking through */
        z-index: 99;
    }
    
    /* Guide Text */
    div[data-testid="stCameraInput"]::before {
        content: "점선 안에 얼굴을 맞춰주세요";
        position: absolute;
        top: 15%;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px black;
        z-index: 100;
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.title("💄 AI 뷰티 어드바이저 (Prototype)")
st.write("얼굴 사진을 올리면 AI가 얼굴형과 이목구비를 분석합니다.")

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Sidebar
st.sidebar.header("설정")
mode = st.sidebar.radio("분석 모드", ["기본 분석 (Face Mesh)", "퍼스널 컬러 (준비중)", "성형 견적 (준비중)"])

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import threading
import time

# --- 📹 Real-time Auto Capture Logic ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.captured_image = None
        self.capture_time = 0
        self.is_aligned = False
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontally for mirror effect
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        aligned = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw Face Mesh on Live Feed
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # 1. Pose Check (Yaw)
                nose_tip_x = face_landmarks.landmark[1].x
                left_ear_x = face_landmarks.landmark[234].x
                right_ear_x = face_landmarks.landmark[454].x
                
                ear_dist = right_ear_x - left_ear_x
                nose_pos = (nose_tip_x - left_ear_x) / ear_dist
                yaw_error = abs(nose_pos - 0.5)
                
                # 2. Center Check
                nose_y = face_landmarks.landmark[1].y
                center_x_error = abs(nose_tip_x - 0.5)
                center_y_error = abs(nose_y - 0.5)
                
                # Criteria: Looking straight (yaw < 0.2) AND Centered (error < 0.3)
                # Relaxed thresholds for better usability
                if yaw_error < 0.2 and center_x_error < 0.3 and center_y_error < 0.3:
                    aligned = True
                    
                    # Draw Green Box to indicate alignment
                    cv2.rectangle(img, (int(w*0.2), int(h*0.1)), (int(w*0.8), int(h*0.9)), (0, 255, 0), 5)
                    
                    if self.capture_time == 0:
                        self.capture_time = time.time()
                        
                    elapsed = time.time() - self.capture_time
                    if elapsed > 1.0:
                        cv2.putText(img, "CAPTURED! CLICK 'ANALYZE'", (50, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        with self.lock:
                            self.captured_image = img_rgb # Save RGB image
                    else:
                        cv2.putText(img, f"Hold still... {1.0-elapsed:.1f}s", (int(w*0.3), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.capture_time = 0 # Reset
                    # Draw Red Box/Guide
                    color = (0, 0, 255)
                    cv2.ellipse(img, (int(w/2), int(h/2)), (int(w*0.25), int(h*0.35)), 0, 0, 360, color, 2)
                    
                    # Debug Info
                    msg = "Adjust Face"
                    if yaw_error >= 0.2: msg = "Look Straight"
                    elif center_x_error >= 0.3: msg = "Center Horizontal"
                    elif center_y_error >= 0.3: msg = "Center Vertical"
                    
                    cv2.putText(img, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(img, f"Yaw: {yaw_error:.2f} (Target < 0.2)", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Always update latest frame for manual capture
        with self.lock:
            self.latest_frame = img_rgb
            self.is_aligned = aligned
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Input Source Selection
input_source = st.radio("이미지 입력 방식", ["사진 업로드", "실시간 자동 촬영 (Beta)"])

# Initialize Session State
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

image = None

if input_source == "사진 업로드":
    st.session_state["captured_image"] = None # Reset capture if switching modes
    uploaded_file = st.file_uploader("얼굴 정면 사진을 올려주세요", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='업로드된 사진', use_column_width=True)

elif input_source == "실시간 자동 촬영 (Beta)":
    st.info("1. 카메라를 켜고 가이드에 얼굴을 맞추세요.\n2. 초록색 박스가 뜨고 'CAPTURED' 메시지가 나오면...\n3. 자동으로 사진이 아래에 뜹니다!")
    
    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Check if image is captured in the processor
    if ctx.video_processor:
        # Auto Capture
        if ctx.video_processor.captured_image is not None:
            if st.session_state["captured_image"] is None:
                st.session_state["captured_image"] = ctx.video_processor.captured_image
                st.rerun()
    
    # Manual Force Capture
    if st.button("📸 지금 화면 캡처하기 (수동)"):
        if ctx.video_processor and hasattr(ctx.video_processor, 'latest_frame'):
            st.session_state["captured_image"] = ctx.video_processor.latest_frame
            st.rerun()

    # Display Captured Image
    if st.session_state["captured_image"] is not None:
        if not st.session_state.get("is_analyzing", False):
            st.success("📸 촬영 성공!")
            st.image(st.session_state["captured_image"], channels="RGB", caption="촬영된 이미지")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 이 사진으로 분석하기"):
                    st.session_state["is_analyzing"] = True
                    st.rerun()
            with col2:
                if st.button("🔄 다시 찍기"):
                    st.session_state["captured_image"] = None
                    st.session_state["is_analyzing"] = False
                    if ctx.video_processor:
                        ctx.video_processor.captured_image = None
                        ctx.video_processor.capture_time = 0
                    st.rerun()
        else:
            # Persist image for analysis
            image = st.session_state["captured_image"]
            
            # Show "Retake" button even during analysis (optional, maybe in sidebar or top)
            if st.sidebar.button("🔄 다른 사진 찍기"):
                st.session_state["captured_image"] = None
                st.session_state["is_analyzing"] = False
                st.rerun()

if image is not None:
    
    st.write("---")
    st.subheader("🔍 AI 분석 중...")

    # Run MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- 🛡️ Quality Control (Pose & Lighting) ---
                h, w, c = image.shape
                
                # 1. Pose Check (Yaw - Looking Left/Right)
                nose_tip_x = face_landmarks.landmark[1].x
                left_ear_x = face_landmarks.landmark[234].x
                right_ear_x = face_landmarks.landmark[454].x
                
                # Calculate relative position of nose between ears
                ear_dist = right_ear_x - left_ear_x
                nose_pos = (nose_tip_x - left_ear_x) / ear_dist # Should be approx 0.5 for frontal
                
                yaw_error = abs(nose_pos - 0.5)
                is_frontal = yaw_error < 0.1 # Allow 10% deviation
                
                # 2. Lighting Check (Brightness)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                is_bright_enough = 80 < brightness < 200 # Not too dark (80), not washed out (200)

                # Display Warnings
                if not is_frontal:
                    st.warning(f"⚠️ 얼굴이 돌아가 있습니다. 정면을 봐주세요. (오차: {yaw_error:.2f})")
                if not is_bright_enough:
                    st.warning(f"⚠️ 조명이 적절하지 않습니다. (밝기: {brightness:.0f}/255). 너무 어둡거나 밝습니다.")

                # Draw landmarks on the image
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                st.image(annotated_image, caption='Face Mesh 분석 결과', use_column_width=True)

                # Basic Analysis Logic (Example)
                h, w, c = image.shape
                
                # Key Landmarks Indices
                # Left Eye: 33, Right Eye: 263, Nose Tip: 1, Chin: 152
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose_tip = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[152]

                # Calculate Distances (Normalized 0-1)
                eye_dist = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
                face_height = np.sqrt((nose_tip.x - chin.x)**2 + (nose_tip.y - chin.y)**2) # Rough approximation

                # --- 🧠 Advanced Analysis Logic (v2.0) ---
                
                # --- 🎨 Personal Color Analysis (Colorwise.me Style) ---
                
                # 1. 🧬 Auto-Extract Colors (Skin, Hair, Eyes)
                # Helper to get average color from a point
                def get_avg_color(img, lm, w, h, offset_y=0):
                    cx, cy = int(lm.x * w), int(lm.y * h) + offset_y
                    # Boundary check
                    cx = max(0, min(cx, w-1))
                    cy = max(0, min(cy, h-1))
                    
                    # Sample 5x5 area
                    roi = img[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
                    if roi.size > 0:
                        return np.mean(roi, axis=(0, 1)).astype(int)
                    return np.array([200, 200, 200]) # Default Grey

                # Skin: Cheek (Landmark 234 is left ear/cheek area, let's move slightly inward to 205)
                skin_color_rgb = get_avg_color(image, face_landmarks.landmark[205], w, h)
                
                # Eyes: Left Iris (Landmark 468)
                eye_color_rgb = get_avg_color(image, face_landmarks.landmark[468], w, h)
                
                # Hair: Top of Forehead (Landmark 10) + Offset Upwards
                # Estimate face height to determine offset
                face_h_est = (face_landmarks.landmark[152].y - face_landmarks.landmark[10].y) * h
                hair_offset = int(-face_h_est * 0.15) # Go up 15% of face height
                hair_color_rgb = get_avg_color(image, face_landmarks.landmark[10], w, h, offset_y=hair_offset)

                # Convert to Hex for Streamlit
                def rgb_to_hex(rgb):
                    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

                skin_hex = rgb_to_hex(skin_color_rgb)
                eye_hex = rgb_to_hex(eye_color_rgb)
                hair_hex = rgb_to_hex(hair_color_rgb)

                # --- 👤 My Color Profile UI ---
                st.divider()
                st.subheader("👤 나의 퍼스널 컬러 프로필 (My Color Profile)")
                st.caption("AI가 분석한 당신의 고유 색상입니다. 실제와 다르다면 눌러서 수정해보세요!")
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    final_skin_hex = st.color_picker("피부색 (Skin)", skin_hex)
                with col_p2:
                    final_eye_hex = st.color_picker("눈동자색 (Eyes)", eye_hex)
                with col_p3:
                    final_hair_hex = st.color_picker("머리색 (Hair)", hair_hex)

                # --- 🧠 Season Prediction (Based on User Profile) ---
                # Convert Final Hex back to RGB for Analysis
                def hex_to_rgb(hex_code):
                    hex_code = hex_code.lstrip('#')
                    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

                analysis_color = hex_to_rgb(final_skin_hex) # Use Skin Color for main season logic
                
                # Convert to LAB for Warm/Cool
                lab_color = cv2.cvtColor(np.uint8([[analysis_color]]), cv2.COLOR_RGB2LAB)[0][0]
                L, A, B = lab_color
                
                # Convert to HSV for Light/Dark
                hsv_color = cv2.cvtColor(np.uint8([[analysis_color]]), cv2.COLOR_RGB2HSV)[0][0]
                H, S, V = hsv_color
                
                # Logic (Simplified):
                is_warm = B > 145
                predicted_season = "분석 불가"
                
                if is_warm:
                    if V > 150:
                        predicted_season = "봄 웜톤 (Spring Warm)"
                        season_desc = "생기 있고 밝은 이미지가 어울립니다."
                    else:
                        predicted_season = "가을 웜톤 (Autumn Warm)"
                        season_desc = "차분하고 깊이 있는 분위기입니다."
                else:
                    if V > 150:
                        predicted_season = "여름 쿨톤 (Summer Cool)"
                        season_desc = "청량하고 맑은 느낌이 베스트입니다."
                    else:
                        predicted_season = "겨울 쿨톤 (Winter Cool)"
                        season_desc = "선명하고 카리스마 있는 스타일입니다."

                # Define Palettes Globally
                SEASON_PALETTES = {
                    "봄 웜톤 (Spring Warm)": [
                        "#FF7F50", "#FFD700", "#98FB98", "#FFA07A", # Coral, Gold, PaleGreen, LightSalmon
                        "#FF6347", "#FFE4B5", "#40E0D0", "#F0E68C"  # Tomato, Moccasin, Turquoise, Khaki
                    ],
                    "여름 쿨톤 (Summer Cool)": [
                        "#FFB6C1", "#E6E6FA", "#87CEFA", "#D8BFD8", # LightPink, Lavender, LightSkyBlue, Thistle
                        "#F0F8FF", "#ADD8E6", "#FFC0CB", "#B0C4DE"  # AliceBlue, LightBlue, Pink, LightSteelBlue
                    ],
                    "가을 웜톤 (Autumn Warm)": [
                        "#8B4513", "#DAA520", "#556B2F", "#CD853F", # SaddleBrown, GoldenRod, Olive, Peru
                        "#A0522D", "#808000", "#D2691E", "#F4A460"  # Sienna, Olive, Chocolate, SandyBrown
                    ],
                    "겨울 쿨톤 (Winter Cool)": [
                        "#DC143C", "#000080", "#FF00FF", "#000000", # Crimson, Navy, Magenta, Black
                        "#FFFFFF", "#4169E1", "#800080", "#2F4F4F"  # White, RoyalBlue, Purple, DarkSlateGray
                    ]
                }

                # --- 🎨 Digital Palette Strip Generation ---
                def create_palette_strip(colors, height=50):
                    num_colors = len(colors)
                    strip_w = 100 * num_colors
                    strip = np.zeros((height, strip_w, 3), dtype=np.uint8)
                    
                    for i, color_hex in enumerate(colors):
                        rgb = hex_to_rgb(color_hex)
                        # CV2 uses BGR
                        bgr = (rgb[2], rgb[1], rgb[0])
                        start_x = i * 100
                        end_x = (i + 1) * 100
                        cv2.rectangle(strip, (start_x, 0), (end_x, height), bgr, -1)
                    
                    return strip

                palette_strip = create_palette_strip(SEASON_PALETTES[predicted_season])

                # 2. 📐 Neoclassical Facial Canons (Golden Ratio)
                # Landmarks
                # Forehead: 10 (Top) -> 168 (Brow)
                # Nose: 168 (Brow) -> 1 (Tip)
                # Chin: 1 (Tip) -> 152 (Bottom)
                
                top_head = face_landmarks.landmark[10]
                mid_brow = face_landmarks.landmark[168]
                nose_tip = face_landmarks.landmark[1]
                chin_bottom = face_landmarks.landmark[152]
                
                # Horizontal Thirds (Vertical Heights)
                forehead_h = np.sqrt((top_head.x - mid_brow.x)**2 + (top_head.y - mid_brow.y)**2)
                nose_h = np.sqrt((mid_brow.x - nose_tip.x)**2 + (mid_brow.y - nose_tip.y)**2)
                chin_h = np.sqrt((nose_tip.x - chin_bottom.x)**2 + (nose_tip.y - chin_bottom.y)**2)
                
                total_h = forehead_h + nose_h + chin_h
                if total_h == 0: total_h = 1
                
                r1 = forehead_h / total_h * 100
                r2 = nose_h / total_h * 100
                r3 = chin_h / total_h * 100
                
                # Vertical Fifths (Horizontal Widths)
                # Left Eye: 33(Outer) - 133(Inner)
                # Inter-Eye: 133(Inner) - 362(Inner)
                # Right Eye: 362(Inner) - 263(Outer)
                
                left_eye_w = np.sqrt((face_landmarks.landmark[33].x - face_landmarks.landmark[133].x)**2 + (face_landmarks.landmark[33].y - face_landmarks.landmark[133].y)**2)
                inter_eye_w = np.sqrt((face_landmarks.landmark[133].x - face_landmarks.landmark[362].x)**2 + (face_landmarks.landmark[133].y - face_landmarks.landmark[362].y)**2)
                right_eye_w = np.sqrt((face_landmarks.landmark[362].x - face_landmarks.landmark[263].x)**2 + (face_landmarks.landmark[362].y - face_landmarks.landmark[263].y)**2)
                
                # Golden Ratio Score (K-Beauty Standard 1:1:0.8)
                # Ideal Proportions:
                # Eyes: 1:1:1 (Inter-eye : Eye Width)
                # Vertical: 1:1:0.8 (Forehead : Nose : Chin) -> Total 2.8
                # Ideal %: Forehead 35.7%, Nose 35.7%, Chin 28.6%
                
                score = 100
                
                # 1. Eye Spacing Penalty (Ideal 1.0)
                eye_ratio = inter_eye_w / left_eye_w
                score -= abs(1.0 - eye_ratio) * 40 
                
                # 2. Vertical Ratio Penalty (Ideal 1:1:0.8)
                # We compare the lower third ratio.
                # Ideal lower third is 0.8 relative to middle third (1.0)
                lower_ratio = chin_h / nose_h if nose_h > 0 else 1.0
                score -= abs(0.8 - lower_ratio) * 50 # Higher penalty for chin ratio deviation
                
                score = max(0, min(100, int(score)))

                # --- 📊 Display Results ---
                st.divider()
                st.subheader("📋 AI 심층 분석 리포트 (K-Beauty Standard)")
                
                # Summary Section (Score & Season)
                col_score, col_season = st.columns(2)
                with col_score:
                    st.markdown(f"### 👑 뷰티 스코어: **{score}점**")
                    st.progress(score)
                    if score >= 90:
                        st.success("상위 1% 황금비율입니다! 🎉")
                    elif score >= 80:
                        st.success("매우 조화로운 비율입니다! ✨")
                    else:
                        st.info("개성 있고 매력적인 비율입니다! 💫")
                
                with col_season:
                    st.markdown(f"### 🎨 퍼스널 컬러: **{predicted_season}**")
                    st.write(season_desc)
                    # Display Palette Strip
                    st.image(palette_strip, caption="✨ 당신의 베스트 컬러 팔레트", use_column_width=True)

                    # --- Data Collection Form (Google Sheets) ---
                    st.divider()
                    st.subheader("💌 결과 저장 및 뉴스레터 구독")
                    st.caption("진단 결과를 저장하고, 더 많은 뷰티 팁을 받아보세요!")

                    with st.form("data_collection_form"):
                        col_form1, col_form2 = st.columns(2)
                        with col_form1:
                            user_name = st.text_input("이름 (Name)")
                        with col_form2:
                            user_email = st.text_input("이메일 (Email)")
                        
                        user_comment = st.text_area("남기고 싶은 말 (선택사항)", placeholder="서비스 이용 후기나 궁금한 점을 적어주세요.")
                        
                        submit_button = st.form_submit_button("💾 결과 저장하기 (Save to Database)")

                        if submit_button:
                            if not user_name or not user_email:
                                st.warning("이름과 이메일을 모두 입력해주세요.")
                            else:
                                try:
                                    # Connect to Google Sheets
                                    conn = st.connection("gsheets", type=GSheetsConnection)
                                    
                                    # Prepare new data
                                    new_data = pd.DataFrame([
                                        {
                                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "Name": user_name,
                                            "Email": user_email,
                                            "Season": predicted_season,
                                            "Best Colors": ", ".join(recommended_colors),
                                            "Comment": user_comment
                                        }
                                    ])
                                    
                                    # Read existing data (to append)
                                    # Note: This might fail if sheet is empty or doesn't exist, handle gracefully
                                    try:
                                        existing_data = conn.read(ttl=0)
                                        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                                    except Exception:
                                        # If read fails (e.g. empty sheet), start with new data
                                        updated_data = new_data
                                    
                                    # Update Sheet
                                    conn.update(data=updated_data)
                                    
                                    st.success("✅ 정보가 성공적으로 저장되었습니다! 감사합니다.")
                                    st.balloons()
                                    
                                except Exception as e:
                                    st.error(f"저장에 실패했습니다. 관리자에게 문의하세요.\nError: {e}")
                                    st.info("※ 배포 환경에서 Google Sheets 연결 설정이 필요합니다.")

                st.divider()
                
                tab1, tab2, tab3 = st.tabs(["🎨 퍼스널 컬러 상세", "📐 황금비율 분석 상세", "💄 가상 메이크업 (Beta)"])
                
                with tab1:
                    st.write("#### 🎨 전 계절 컬러 비교 (All Seasons)")
                    st.caption("AI 예측이 틀릴 수도 있습니다. 다른 계절의 색상도 직접 대보며 가장 잘 어울리는 톤을 찾아보세요!")
                    
                    # Season Selector
                    selected_season = st.radio(
                        "확인하고 싶은 계절을 선택하세요:",
                        list(SEASON_PALETTES.keys()),
                        index=list(SEASON_PALETTES.keys()).index(predicted_season) if predicted_season in SEASON_PALETTES else 0,
                        horizontal=True
                    )
                    
                    # Season Descriptions
                    SEASON_DESCRIPTIONS = {
                        "봄 웜톤 (Spring Warm)": "생기 있고 밝은 이미지가 어울립니다. (Best: 코랄, 피치, 옐로우)",
                        "여름 쿨톤 (Summer Cool)": "청량하고 맑은 느낌이 베스트입니다. (Best: 파스텔 핑크, 스카이블루)",
                        "가을 웜톤 (Autumn Warm)": "차분하고 깊이 있는 분위기입니다. (Best: 브라운, 카키, 머스타드)",
                        "겨울 쿨톤 (Winter Cool)": "선명하고 카리스마 있는 스타일입니다. (Best: 블랙, 화이트, 비비드)"
                    }
                    
                    current_palette = SEASON_PALETTES[selected_season]
                    current_desc = SEASON_DESCRIPTIONS.get(selected_season, "")
                    
                    st.write(f"#### 👗 {selected_season} 배경 매칭")
                    st.info(f"💡 **{selected_season} 특징:** {current_desc}")
                    
                    # --- 🛍️ Styling Guide ---
                    SEASON_TIPS = {
                        "봄 웜톤 (Spring Warm)": {
                            "Fashion": "따뜻하고 밝은 파스텔 톤이나 비비드한 컬러가 잘 어울립니다. (코랄, 피치, 개나리색)",
                            "Makeup": "복숭아빛 블러셔와 코랄/오렌지 립이 베스트! 펄은 골드 펄을 추천해요.",
                            "Hair": "밝은 갈색, 오렌지 브라운, 골드 브라운 등 따뜻한 계열의 염색이 화사해 보입니다.",
                            "Jewelry": "실버보다는 **골드**나 로즈골드 액세서리가 피부와 잘 어우러집니다."
                        },
                        "여름 쿨톤 (Summer Cool)": {
                            "Fashion": "흰끼가 섞인 파스텔 톤이나 차분한 그레이시 컬러가 우아함을 더해줍니다. (라벤더, 스카이블루)",
                            "Makeup": "딸기우유 핑크, 라벤더 블러셔가 찰떡! 립은 핑크나 플럼 계열을 추천해요.",
                            "Hair": "자연모(흑발)나 애쉬 브라운, 초코 브라운처럼 붉은기가 없는 차분한 색이 좋습니다.",
                            "Jewelry": "골드보다는 **실버**나 화이트골드, 진주 액세서리가 깨끗한 이미지를 줍니다."
                        },
                        "가을 웜톤 (Autumn Warm)": {
                            "Fashion": "깊이 있고 차분한 어스(Earth) 컬러가 분위기 여신으로 만들어줍니다. (카키, 머스타드, 벽돌색)",
                            "Makeup": "음영 메이크업이 가장 잘 어울려요. 말린 장미(MLBB), 브릭 레드 립을 시도해보세요.",
                            "Hair": "다크 브라운, 카푸치노 브라운 등 깊고 풍성한 컬러가 고급스러워 보입니다.",
                            "Jewelry": "광택이 적은 **앤틱 골드**나 브론즈, 우드 소재의 액세서리가 멋스럽습니다."
                        },
                        "겨울 쿨톤 (Winter Cool)": {
                            "Fashion": "선명하고 대비가 확실한 컬러가 카리스마를 살려줍니다. (블랙&화이트, 로얄 블루, 핫핑크)",
                            "Makeup": "아이라이너를 깔끔하게 그리고, 레드나 푸시아 핑크 립으로 포인트를 주세요.",
                            "Hair": "윤기 나는 흑발(블루 블랙)이 가장 베스트! 애매한 갈색보다는 확실한 블랙이 낫습니다.",
                            "Jewelry": "반짝이는 **실버**, 화이트골드, 다이아몬드처럼 화려하고 차가운 느낌이 잘 어울립니다."
                        }
                    }
                    
                    tips = SEASON_TIPS.get(selected_season, {})
                    
                    with st.expander(f"🛍️ {selected_season} 스타일링 가이드 (클릭)", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**👚 패션 (Fashion)**\n- {tips['Fashion']}")
                            st.markdown(f"**💍 주얼리 (Jewelry)**\n- {tips['Jewelry']}")
                        with c2:
                            st.markdown(f"**💄 메이크업 (Makeup)**\n- {tips['Makeup']}")
                            st.markdown(f"**💇‍♀️ 헤어 (Hair)**\n- {tips['Hair']}")
                    
                    # Single Color Selection for Detail View
                    st.write("👇 **상세 컬러 선택 (클릭하여 변경)**")
                    selected_color = st.radio(
                        "테스트할 색상을 선택하세요:",
                        current_palette,
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    
                    def apply_background(img, landmarks, hex_color):
                        # Hex to BGR
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        color_bgr = (b, g, r)
                        
                        h, w, c = img.shape
                        
                        # Jawline Indices (Left Ear -> Chin -> Right Ear)
                        jawline_indices = [
                            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 
                            377, 400, 378, 379, 365, 397, 288, 361, 323, 454
                        ]
                        
                        points = []
                        for idx in jawline_indices:
                            pt = landmarks.landmark[idx]
                            points.append((int(pt.x * w), int(pt.y * h)))
                        
                        # Add bottom corners to create a "bib" or "clothing" shape
                        points.append((w, h)) # Bottom Right
                        points.append((0, h)) # Bottom Left
                        
                        points = np.array(points, np.int32)
                        
                        # Create Mask
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 255)
                        
                        # Soften edges
                        mask = cv2.GaussianBlur(mask, (15, 15), 0)
                        
                        # Create Color Layer
                        color_layer = np.full((h, w, 3), color_bgr, dtype=np.uint8)
                        
                        # Combine: Original Image + Color Layer (masked)
                        # We want the color to be ON the mask (Neck/Chest), and original image elsewhere.
                        mask_norm = mask.astype(float) / 255.0
                        mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
                        
                        # Output = Color * Mask + Original * (1 - Mask)
                        out = (color_layer.astype(float) * mask_norm + img.astype(float) * (1.0 - mask_norm)).astype(np.uint8)
                        
                        return out

                    # Display Large Preview
                    if selected_color:
                        large_bg_img = apply_background(image, face_landmarks, selected_color)
                        st.image(large_bg_img, caption=f"선택된 컬러: {selected_color}", use_column_width=True)
                    
                    # Display Palette Grid (Small)
                    with st.expander("🎨 전체 팔레트 모아보기 (클릭해서 펼치기)"):
                        cols1 = st.columns(4)
                        for i in range(4):
                            with cols1[i]:
                                bg_img = apply_background(image, face_landmarks, current_palette[i])
                                st.image(bg_img, caption=current_palette[i], use_column_width=True)
                        
                        cols2 = st.columns(4)
                        for i in range(4):
                            with cols2[i]:
                                bg_img = apply_background(image, face_landmarks, current_palette[i+4])
                                st.image(bg_img, caption=current_palette[i+4], use_column_width=True)
                
                with tab2:
                    st.write("#### 1. 얼굴 세로 비율 (트렌드 1:1:0.8)")
                    st.caption("최신 한국 미인상은 하안부(턱)가 짧은 '동안 비율'을 선호합니다.")
                    
                    # Normalize to Middle Third = 1.0
                    if nose_h > 0:
                        r_top = forehead_h / nose_h
                        r_mid = 1.0
                        r_bot = chin_h / nose_h
                    else:
                        r_top, r_mid, r_bot = 1.0, 1.0, 1.0
                        
                    st.write(f"- 상안부(이마): **{r_top:.2f}**")
                    st.write(f"- 중안부(코): **{r_mid:.2f}**")
                    st.write(f"- 하안부(턱): **{r_bot:.2f}** (이상적 0.8)")
                    
                    if 0.75 <= r_bot <= 0.85:
                        st.success("✨ 완벽한 '동안(Baby Face)' 비율입니다! (1:1:0.8)")
                    elif r_bot < 0.75:
                        st.info("💡 하안부가 매우 짧아 귀여운 이미지입니다.")
                    else:
                        st.info("💡 하안부가 긴 편으로, 성숙하고 우아한 '배우상' 이미지입니다.")

                    st.markdown("---")
                    st.write("#### 2. 눈 비율 (이상적 1:1:1)")
                    st.caption(f"눈 너비 : 미간 너비 = 1 : {eye_ratio:.2f}")
                    
                    if 0.9 <= eye_ratio <= 1.1:
                        st.success("눈과 미간의 비율이 황금비율(1:1)에 완벽하게 부합합니다!")
                    elif eye_ratio > 1.1:
                        st.warning("미간이 눈보다 넓습니다. (앞트임 메이크업 추천)")
                    else:
                        st.warning("미간이 눈보다 좁습니다. (뒤트임/밑트임 메이크업 추천)")
                
                with tab3:
                    st.write("#### 💄 가상 메이크업 (Virtual Makeover)")
                    st.info("원하는 스타일을 선택하여 내 얼굴에 직접 적용해보세요!")
                    
                    makeover_img = image.copy()
                    
                    # 1. Virtual Lipstick
                    st.markdown("##### 💋 립스틱 (Lipstick)")
                    lip_color = st.color_picker("립스틱 색상 선택", "#FF0055")
                    lip_opacity = st.slider("진하기 (Opacity)", 0.0, 1.0, 0.4)
                    
                    def apply_lipstick(img, landmarks, hex_color, opacity):
                        # Hex to RGB
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        color_rgb = (r, g, b)
                        
                        h, w, c = img.shape
                        
                        # Lip Indices (Outer)
                        lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
                        
                        points = []
                        for idx in lip_indices:
                            pt = landmarks.landmark[idx]
                            points.append((int(pt.x * w), int(pt.y * h)))
                        points = np.array(points, np.int32)
                        
                        # Create Mask
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 255)
                        mask = cv2.GaussianBlur(mask, (7, 7), 0) # Soft edges
                        
                        # Create Color Layer
                        color_layer = np.full((h, w, 3), color_rgb, dtype=np.uint8)
                        
                        # Blend
                        mask_norm = (mask.astype(float) / 255.0) * opacity
                        mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
                        
                        out = (color_layer.astype(float) * mask_norm + img.astype(float) * (1.0 - mask_norm)).astype(np.uint8)
                        return out

                    makeover_img = apply_lipstick(makeover_img, face_landmarks, lip_color, lip_opacity)
                    
                    # 2. Virtual Hair Dye (Beta) - Improved
                    st.markdown("##### 💇‍♀️ 헤어 염색 (Hair Dye) - Beta")
                    
                    dye_mode = st.radio("염색 영역 선택 방식", ["자동 (Auto)", "수동 그리기 (Manual Draw)"], horizontal=True)
                    
                    dye_color = st.color_picker("염색할 색상 선택", "#8B4513")
                    dye_intensity = st.slider("염색 강도", 0.0, 1.0, 0.5)

                    def apply_hair_dye(img, seed_hex, target_hex, intensity, tolerance, landmarks, skin_rgb=None, correction_mask=None, return_mask=False):
                        h, w, c = img.shape
                        
                        # 1. Color Threshold Mask (HSV)
                        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                        
                        # Seed Color (Source)
                        seed_hex = seed_hex.lstrip('#')
                        sr, sg, sb = tuple(int(seed_hex[i:i+2], 16) for i in (0, 2, 4))
                        seed_bgr = np.uint8([[[sb, sg, sr]]])
                        seed_hsv = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2HSV)[0][0]
                        
                        lower_bound = np.array([max(0, seed_hsv[0] - tolerance), 20, 20]) 
                        upper_bound = np.array([min(179, seed_hsv[0] + tolerance), 255, 255])
                        
                        color_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
                        
                        # 2. Selfie Segmentation (Exclude Background)
                        mp_selfie_segmentation = mp.solutions.selfie_segmentation
                        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
                            res = selfie_seg.process(img)
                            # condition: > 0.5 is person
                            person_mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
                        
                        # 3. Face Exclusion Mask (Protect Face Skin)
                        face_mask = np.zeros((h, w), dtype=np.uint8)
                        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                        
                        face_points = []
                        for idx in face_oval_indices:
                            pt = landmarks.landmark[idx]
                            face_points.append((int(pt.x * w), int(pt.y * h)))
                        
                        if face_points:
                            cv2.fillPoly(face_mask, [np.array(face_points)], 255)
                            face_mask = cv2.dilate(face_mask, np.ones((15,15), np.uint8), iterations=1)

                        # 4. Skin Color Exclusion (Protect Neck/Body Skin)
                        skin_exclusion_mask = np.zeros((h, w), dtype=np.uint8)
                        if skin_rgb is not None:
                            skin_bgr = np.uint8([[[skin_rgb[2], skin_rgb[1], skin_rgb[0]]]])
                            skin_hsv = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2HSV)[0][0]
                            
                            # Wide range for skin to catch shadows on neck
                            s_lower = np.array([max(0, skin_hsv[0] - 20), 30, 30])
                            s_upper = np.array([min(179, skin_hsv[0] + 20), 255, 255])
                            
                            skin_exclusion_mask = cv2.inRange(hsv_img, s_lower, s_upper)
                            skin_exclusion_mask = cv2.dilate(skin_exclusion_mask, np.ones((5,5), np.uint8), iterations=2)

                        # 5. Combine Masks
                        # Hair = (Color Match) AND (Person) AND (NOT Face) AND (NOT Skin)
                        final_mask = cv2.bitwise_and(color_mask, person_mask)
                        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(face_mask))
                        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(skin_exclusion_mask))
                        
                        # 6. Apply Correction Mask (User Add/Remove)
                        if correction_mask is not None:
                            # Green (Channel 1) = ADD
                            mask_add = correction_mask[:, :, 1]
                            # Red (Channel 0) = REMOVE
                            mask_remove = correction_mask[:, :, 0]
                            
                            # Add first
                            final_mask = cv2.bitwise_or(final_mask, mask_add)
                            # Then Remove
                            final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(mask_remove))

                        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

                        if return_mask:
                            # Return mask as RGB image for visualization
                            return cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)

                        # Apply Color (LAB Blending for Natural Look)
                        target_hex = target_hex.lstrip('#')
                        tr, tg, tb = tuple(int(target_hex[i:i+2], 16) for i in (0, 2, 4))
                        target_rgb = np.uint8([[[tr, tg, tb]]])
                        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0][0]
                        t_l, t_a, t_b = int(target_lab[0]), int(target_lab[1]), int(target_lab[2])

                        # Convert Image to LAB
                        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(img_lab)

                        # Prepare Mask
                        mask_f = final_mask.astype(float) / 255.0 * intensity
                        
                        # Blend A and B channels (Color)
                        a_new = (a.astype(float) * (1.0 - mask_f) + t_a * mask_f).astype(np.uint8)
                        b_new = (b.astype(float) * (1.0 - mask_f) + t_b * mask_f).astype(np.uint8)
                        
                        # Blend L channel slightly (Optional: 20% influence) to allow some lightness change but keep texture
                        # Too much L blending kills texture. 0.2 is safe.
                        l_new = (l.astype(float) * (1.0 - mask_f * 0.3) + t_l * (mask_f * 0.3)).astype(np.uint8)

                        # Merge and Convert back
                        out_lab = cv2.merge([l_new, a_new, b_new])
                        out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
                        
                        return out

                    if dye_mode == "자동 (Auto)":
                        st.caption("※ '내 머리색 지정'을 조절하여 염색될 영역을 선택하세요.")
                        c_h1, c_h2 = st.columns(2)
                        with c_h1:
                            # Default to the auto-detected hair color
                            def rgb_to_hex(rgb): return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
                            default_hair_hex = rgb_to_hex(hair_color_rgb)
                            ref_hair_color = st.color_picker("내 머리색 지정 (Source)", default_hair_hex, help="이 색상과 비슷한 영역이 염색됩니다.")
                        with c_h2:
                            color_tolerance = st.slider("색상 인식 범위 (Tolerance)", 10, 150, 50)
                        
                        show_mask = st.checkbox("🧐 염색 영역(마스크) 미리보기")
                        
                        # Correction Tool
                        use_correction = st.checkbox("🛠️ 영역 수정 (추가/제거)")
                        correction_mask_full = None
                        
                        if use_correction:
                            st.caption("👇 '추가'는 초록색, '제거'는 빨간색으로 칠해집니다.")
                            
                            # Resize for Canvas
                            canvas_width = 600
                            aspect_ratio = image.shape[0] / image.shape[1]
                            canvas_height = int(canvas_width * aspect_ratio)
                            
                            # Ensure image is uint8 and RGB for PIL
                            if image.dtype != np.uint8:
                                image_u8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                            else:
                                image_u8 = image
                                
                            img_pil = Image.fromarray(image_u8).convert("RGB")
                            img_resized = img_pil.resize((canvas_width, canvas_height))
                            
                            col_tool1, col_tool2 = st.columns([1, 2])
                            with col_tool1:
                                brush_mode = st.radio("브러쉬 모드", ["추가 (Add)", "제거 (Remove)"])
                            with col_tool2:
                                stroke_width = st.slider("브러쉬 크기", 1, 50, 20)
                                
                            if brush_mode == "추가 (Add)":
                                stroke_color = "#00FF00" # Green
                                fill_color = "rgba(0, 255, 0, 0.3)"
                            else:
                                stroke_color = "#FF0000" # Red
                                fill_color = "rgba(255, 0, 0, 0.3)"
                            
                            canvas_result = st_canvas(
                                fill_color=fill_color,
                                stroke_width=stroke_width,
                                stroke_color=stroke_color,
                                background_image=img_resized,
                                update_streamlit=True,
                                height=canvas_height,
                                width=canvas_width,
                                drawing_mode="freedraw",
                                key="correction_canvas_v2",
                            )
                            
                            if canvas_result.image_data is not None:
                                mask_resized = canvas_result.image_data # RGBA
                                if np.sum(mask_resized) > 0:
                                    correction_mask_full = cv2.resize(mask_resized, (image.shape[1], image.shape[0]))
                        
                        # Apply Auto Dye (Existing Function)
                        makeover_img = apply_hair_dye(makeover_img, ref_hair_color, dye_color, dye_intensity, color_tolerance, face_landmarks, skin_rgb=skin_color_rgb, correction_mask=correction_mask_full, return_mask=show_mask)
                        st.image(makeover_img, caption="✨ 메이크업 & 염색 적용 결과" if not show_mask else "🧐 염색 영역 마스크 (흰색 부분이 염색됨)", use_column_width=True)

                    else: # Manual Draw Mode
                        st.caption("👇 사진 위에 염색하고 싶은 부위를 직접 칠해주세요!")
                        
                        # 1. Resize for Canvas (to fit screen)
                        canvas_width = 600
                        aspect_ratio = image.shape[0] / image.shape[1]
                        canvas_height = int(canvas_width * aspect_ratio)
                        
                        img_pil = Image.fromarray(image)
                        img_resized = img_pil.resize((canvas_width, canvas_height))
                        
                        # Stroke width
                        stroke_width = st.slider("브러쉬 크기", 1, 50, 20)
                        
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=stroke_width,
                            stroke_color="#ffffff",
                            background_image=img_resized,
                            update_streamlit=True,
                            height=canvas_height,
                            width=canvas_width,
                            drawing_mode="freedraw",
                            key="canvas",
                        )
                        
                        if canvas_result.image_data is not None:
                            # Get the drawn mask (Alpha channel) from the resized canvas
                            mask_resized = canvas_result.image_data[:, :, 3]
                            
                            if np.sum(mask_resized) > 0:
                                # 2. Scale Mask back to Original Size
                                mask = cv2.resize(mask_resized, (image.shape[1], image.shape[0]))
                                
                                # Apply Dye using Manual Mask (LAB Blending)
                                h, w, c = image.shape
                                
                                # Target Color LAB
                                target_hex = dye_color.lstrip('#')
                                tr, tg, tb = tuple(int(target_hex[i:i+2], 16) for i in (0, 2, 4))
                                target_rgb = np.uint8([[[tr, tg, tb]]])
                                target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)[0][0]
                                t_l, t_a, t_b = int(target_lab[0]), int(target_lab[1]), int(target_lab[2])
                                
                                # Image LAB
                                img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                                l, a, b = cv2.split(img_lab)
                                
                                # Mask
                                mask_f = (mask.astype(float) / 255.0) * dye_intensity
                                
                                # Blend
                                a_new = (a.astype(float) * (1.0 - mask_f) + t_a * mask_f).astype(np.uint8)
                                b_new = (b.astype(float) * (1.0 - mask_f) + t_b * mask_f).astype(np.uint8)
                                l_new = (l.astype(float) * (1.0 - mask_f * 0.3) + t_l * (mask_f * 0.3)).astype(np.uint8)
                                
                                out_lab = cv2.merge([l_new, a_new, b_new])
                                out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
                                
                                # Apply previous makeover effects (Lipstick) if any? 
                                # Note: 'makeover_img' has lipstick. 'image' is original.
                                # If we want to stack, we should use 'makeover_img' as base.
                                # Let's use makeover_img as base to keep lipstick.
                                
                                img_lab_base = cv2.cvtColor(makeover_img, cv2.COLOR_RGB2LAB)
                                l_base, a_base, b_base = cv2.split(img_lab_base)
                                
                                a_final = (a_base.astype(float) * (1.0 - mask_f) + t_a * mask_f).astype(np.uint8)
                                b_final = (b_base.astype(float) * (1.0 - mask_f) + t_b * mask_f).astype(np.uint8)
                                l_final = (l_base.astype(float) * (1.0 - mask_f * 0.3) + t_l * (mask_f * 0.3)).astype(np.uint8)
                                
                                out_lab_final = cv2.merge([l_final, a_final, b_final])
                                out = cv2.cvtColor(out_lab_final, cv2.COLOR_LAB2RGB)
                                
                                st.image(out, caption="✨ 수동 염색 적용 결과", use_column_width=True)
                            else:
                                st.info("👆 위 사진에 염색할 부위를 칠해보세요.")

                st.warning("⚠️ 이 결과는 AI의 추정치이며, 조명과 각도에 따라 달라질 수 있습니다.")
else:
    st.info("사진을 업로드하면 분석이 시작됩니다.")
