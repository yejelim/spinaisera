import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison
import pandas as pd

# 사용자 지정 CSS 추가
st.markdown("""
    <style>
        .stMarkdown h3 {
            color: #007ACC;
        }
        .stMarkdown p {
            font-size: 16px;
        }
        .stMarkdown {
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <meta name="description" content="SERA는 척추 내시경 수술을 지원하는 AI 기반 시스템으로, 실시간 출혈 감지 및 수술 효율성을 높이는 혁신적인 기술을 제공합니다.">
    <meta name="keywords" content="Streamlit, SERA, 척추 내시경, 출혈 감지, 의료 AI, 실시간 영상 분석, SPINAI, spine endoscopy">
    <meta property="og:title" content="SERA Program | Spine Endoscopy Robotic API">
    <meta property="og:description" content="SERA는 척추 내시경 수술에서 출혈 위치를 실시간 감지하고, 수술 효율성을 극대화하는 AI 기반 시스템입니다.">
    <meta property="og:image" content="https://spinai-sera.streamlit.app/logo.png">
    <meta property="og:url" content="https://spinai-sera.streamlit.app/">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="SERA Program | 척추 내시경 수술 지원 AI">
    <meta name="twitter:description" content="SERA는 척추 내시경 수술에서 출혈 위치를 실시간 감지하고, 수술 효율성을 극대화하는 AI 기반 시스템입니다.">
    """,
    unsafe_allow_html=True
)


# 언어 선택 탭
selected_language = st.sidebar.selectbox("Select Language", ["English", "Korean"])

# 데이터 준비
def get_performance_data():
    performance_data = {
        "YOLO Type": ["YOLO-V11 n", "YOLO-V11 s", "YOLO-V11 m", "YOLO-V11 l", "YOLO-V11 x"],
        "Precision (B)": [0.79856, 0.84071, 0.86423, 0.83328, 0.87132],
        "Recall (B)": [0.70024, 0.72766, 0.73339, 0.76306, 0.73108],
        "mAP50 (B)": [0.73942, 0.78281, 0.79353, 0.7948, 0.79397],
        "mAP50-95 (B)": [0.5572, 0.61227, 0.63615, 0.63868, 0.64306],
        "Precision (M)": [0.79265, 0.84034, 0.86172, 0.82579, 0.87278],
        "Recall (M)": [0.69716, 0.72767, 0.73205, 0.75477, 0.72716],
        "mAP50 (M)": [0.72748, 0.7745, 0.78648, 0.78336, 0.78339],
        "mAP50-95 (M)": [0.51671, 0.56514, 0.5878, 0.58784, 0.5909],
        "Image Segmentation Time on NVIDIA RTX 4090 (s)": [0.0128, 0.0124, 0.0142, 0.0167, 0.0200],
    }
    df = pd.DataFrame(performance_data)
    return df

# 데이터 수정 및 강조 표시
def style_dataframe(df):
    df_styled = df.style.applymap(
        lambda x: "color: red; font-weight: bold;" if x in [0.87278, 0.0200] else ""
    )
    return df_styled

def add_padding(image, padding, color=(0, 0, 0)):
    new_width = image.width
    new_height = image.height + padding * 2
    new_image = Image.new("RGB", (new_width, new_height), color)
    new_image.paste(image, (0, padding))
    return new_image

def resize_to_same_height(img1, img2, target_height):
    w1, h1 = img1.size
    w2, h2 = img2.size

    new_w1 = int((target_height / h1) * w1)
    new_w2 = int((target_height / h2) * w2)

    img1_resized = img1.resize((new_w1, target_height))
    img2_resized = img2.resize((new_w2, target_height))

    return img1_resized, img2_resized

if selected_language == "English":
    st.image('./logo.png', width=200)  # SPiNAI Logo
    st.title("SERA Program Introduction")

    st.markdown("""
    <h3>
    <b>S</b>pine <b>E</b>ndoscopy <b>R</b>obotic <b>A</b>PI - <b>SERA</b>
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("""
    SERA is an AI-based system developed to assist spinal endoscopic surgery,
    detecting bleeding locations in real-time and improving surgical efficiency through innovative technology.
    """)

    st.header("Main Objectives of SERA")
    st.markdown("""
    - **Real-time bleeding location detection and hemostasis assistance**: Improve visibility during surgery affected by bleeding.
    - **Development of foundational technology for the robotic era**: Integration possibilities with robotic assistance in spinal endoscopic surgery.
    - **Providing medical education materials**: Combining with video generation AI like SORA to create educational content.
    """)

    st.header("SERA Demo Video")
    video_file = open('./demovideo.mp4', "rb")
    st.video(video_file.read())

    st.header("SERA Results Comparison")
    # 반응형 화면 크기에 따라 max_width 설정
    def get_max_width():
        # Streamlit에서 기본적으로 화면 크기를 직접 가져올 수 없으므로 임시 설정
        return 700  # 기본 최대 너비

    # 이미지 로드 및 크기 조정
    image1 = Image.open('image1.jpg')
    image2 = Image.open('image2.png')

    max_width = get_max_width()

    # 이미지 크기 비율 계산 및 조정
    ratio1 = min(1, max_width / image1.width)  # 화면 크기에 맞게 비율 조정
    ratio2 = min(1, max_width / image2.width)

    image1_resized = image1.resize((int(image1.width * ratio1), int(image1.height * ratio1)))
    image2_resized = image2.resize((int(image2.width * ratio2), int(image2.height * ratio2)))

    # 패딩 추가
    def add_padding(image, padding, color=(0, 0, 0)):
        new_width = image.width
        new_height = image.height + padding * 2
        new_image = Image.new("RGB", (new_width, new_height), color)
        new_image.paste(image, (0, padding))
        return new_image

    padding = 20
    image1_padded = add_padding(image1_resized, padding, color=(0, 0, 0))
    image2_padded = add_padding(image2_resized, padding, color=(0, 0, 0))

    # Streamlit CSS를 통해 반응형 레이아웃 추가
    st.markdown(
        """
        <style>
        .comparison-container {
            max-width: 100%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 이미지 비교 표시
    st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
    image_comparison(
        img1=image1_padded,
        img2=image2_padded,
        label1="Original Image",
        label2="SERA Image",
        width=max_width
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.header("Technical Details")
    st.markdown("""
    - **YOLO-V11 Model**: Optimized deep learning network for processing spinal endoscopic images in real-time.
    - **Dataset**: 3,054 frames collected from 6 patients.
      - Annotated with 9 classes (Bleeding Focus, Vessel, Instrument, Bone, Ligamentum flavum, etc.).
    - **Data Augmentation**: Flipping, rotation, Gaussian noise, and brightness adjustments.
    - **Training**: 300 epochs across 5 models (n, s, m, l, x).
    """)

    st.header("Model Results and Performance")
    df = get_performance_data()
    styled_df = style_dataframe(df)
    st.dataframe(styled_df)

    st.header("Future Directions")
    st.markdown("""
    - **Real-time surgical support**: Minimize complications like bleeding with real-time data processing.
    - **Integration with robotic surgery**: Potential collaboration with robotic systems from companies like Medtronic and Stryker.
    - **Enhanced accuracy**: Expecting improved performance with additional data and model refinement.
    """)

    st.markdown("""
        <div style="padding: 10px; border: 1px solid #007ACC; border-radius: 10px; background-color: #f9f9f9;">
            <h4>Contact</h4>
            <p><b>Hyun-Ji Jung</b><br>
            School of Medicine, Pusan National University<br>
            <a href="mailto:hgi0629@pusan.ac.kr">hgi0629@pusan.ac.kr</a></p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.image('./logo.png', width=200)  # SPiNAI 로고
    st.title("SERA 프로그램 소개")

    st.markdown("""
    <h3>
    <b>S</b>pine <b>E</b>ndoscopy <b>R</b>obotic <b>A</b>PI - <b>SERA</b>
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("""
    SERA는 척추 내시경 수술을 지원하기 위해 개발된 AI 기반 시스템입니다. 
    SERA는 실시간으로 출혈 위치를 감지하고, 척추 내시경 수술 중 수술 효율성을 극대화하는 혁신적인 의료 AI 기술입니다.

    """)

    st.header("SERA의 주요 목표")
    st.markdown("""
    - **실시간 출혈 위치 감지 및 지혈 보조**: 수술 중 출혈로 인해 시야가 가려지는 상황을 개선.
    - **로봇 시대를 위한 기반 기술 개발**: 척추 내시경 수술에서 로봇 보조 시스템과의 통합 가능성.
    - **의료 교육 자료 제공**: SORA와 같은 비디오 생성 AI와 결합하여 교육 콘텐츠 제작.
    """)

    st.header("SERA 데모 비디오")
    video_file = open('./demovideo.mp4', "rb")
    st.video(video_file.read())

    st.header("SERA 결과 비교")
    image1 = Image.open('image1.jpg')
    image2 = Image.open('image2.png')

    max_width = 700
    ratio1 = max_width / image1.width
    ratio2 = max_width / image2.width

    image1_resized = image1.resize((int(image1.width * ratio1), int(image1.height * ratio1)))
    image2_resized = image2.resize((int(image2.width * ratio2), int(image2.height * ratio2)))

    padding = 20
    image1_padded = add_padding(image1_resized, padding, color=(0, 0, 0))
    image2_padded = add_padding(image2_resized, padding, color=(0, 0, 0))

    image_comparison(
        img1=image1_padded,
        img2=image2_padded,
        label1="Original Image",
        label2="SERA Image",
        width=max_width
    )

    st.header("기술 세부사항")
    st.markdown("""
    - **YOLO-V11 모델 사용**: 최적화된 딥러닝 네트워크를 활용하여 실시간으로 척추 내시경 영상을 처리.
    - **데이터셋**: 6명의 환자에서 수집된 3,054개의 프레임을 사용.
      - 9개의 클래스 (Bleeding Focus, Vessel, instrument, bone, Ligamentum flavum 등)로 주석 처리.
    - **데이터 증강**: 플리핑, 회전, 가우시안 노이즈 및 밝기 조정을 사용.
    - **훈련**: 5가지 모델(n, s, m, l, x)로 300 epochs 동안 학습.
    """)

    st.header("모델 결과 및 성능")
    df = get_performance_data()
    styled_df = style_dataframe(df)
    st.dataframe(styled_df)

    st.header("미래 방향")
    st.markdown("""
    - **실시간 수술 지원**: 실시간 데이터 처리로 출혈과 같은 수술 합병증 최소화.
    - **로봇 수술과의 통합**: Medtronic, Stryker 사 척추 수술 로봇과의 결합 가능성.
    - **정확도 향상**: 추가 데이터와 모델 개선으로 더욱 향상된 성능 기대.
    """)

    st.markdown("""
        <div style="padding: 10px; border: 1px solid #007ACC; border-radius: 10px; background-color: #f9f9f9;">
            <h4>문의</h4>
            <p><b>Hyun-Ji Jung, 정현지</b><br>
            School of Medicine, Pusan National University<br>
            <a href="mailto:hgi0629@pusan.ac.kr">hgi0629@pusan.ac.kr</a></p>
        </div>
    """, unsafe_allow_html=True)
