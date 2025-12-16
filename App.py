import io
import math
import zipfile
from PIL import Image, ImageOps

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0

from model import (
    init_sd,
    generate_images,
    init_classifier,
    score_image_pil,
    device, 
)


st.set_page_config(page_title="AI Fashion Outfit Recommender", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Noto Sans JP', sans-serif;
        background-color: #fafafa;
    }
    .block-container { padding-top: 1.2rem; }

    .hero {
        padding: 1.4rem 1.8rem;
        border-radius: 20px;
        background: #0f172a;
        color: white;
        margin-bottom: 1.5rem;
    }

    .card {
        padding: 1.4rem;
        border-radius: 20px;
        background: white;
        box-shadow: 0 12px 28px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
    }

    .stButton > button {
        border-radius: 14px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>AI Fashion Outfit Recommender</h1>
        <p>Upload one item. Discover AI-curated outfit recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    with st.expander("Model Settings", expanded=False):
        st.info("モデルファイルはGitHubリポジトリの相対パスに配置してください")
        
        pipe_path = st.text_input("SD model path", "")
        
        lora_path = st.text_input(
            "LoRA directory path",
            ".", 
        )
        lora_weight_name = st.text_input(
            "LoRA weight file",
            "zemi_notXL-10.safetensors",
        )
        clf_ckpt_path = st.text_input(
            "Classifier checkpoint file",
            "fashion_model.pth",
        )

        if st.button("Initialize Models", use_container_width=True):
            with st.spinner("Initializing models... (This may take several minutes on CPU)"):
                try:
                    init_sd(
                        pipe_path=pipe_path if pipe_path else None,
                        lora_path=lora_path if lora_path != "." else None,
                        lora_weight_name=lora_weight_name,
                    )
                except Exception as e:
                    st.error(f"Failed to initialize Stable Diffusion Pipeline: {e}")
                    st.stop()
                    
                try:
                    model = efficientnet_b0(pretrained=False)
                    model.classifier[1] = torch.nn.Linear(
                        model.classifier[1].in_features, 2
                    )
                    state = torch.load(clf_ckpt_path, map_location="cpu")
                    model.load_state_dict(state)
                    model.eval()
                    model.to(device)

                    transform = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225],
                            ),
                        ]
                    )
                    init_classifier(model, transform)
                except FileNotFoundError:
                    st.error(f"Classifier checkpoint not found at: {clf_ckpt_path}")
                    st.stop()
                except Exception as e:
                    st.error(f"Failed to initialize Classifier: {e}")
                    st.stop()


            st.session_state["initialized"] = True
            st.success("Models initialized successfully!")
            st.rerun() 

    with st.expander("Generation Settings", expanded=True):
        prompt_text = st.text_input(
            "Style hint",
            "a full-body photo of a stylish man wearing the uploaded clothing item, casual fashion, realistic lighting, high quality, neutral background",
        )
        total_images = st.slider("Candidates", 6, 60, 24, step=2)
        batch_size = st.select_slider(
            "Batch Size (Speed vs Stability)",
            options=[1, 2, 4, 6, 8],
            value=4,
        )
        resize_to_512 = st.checkbox("Resize input to 512×512", True)
        seed = st.number_input("Seed (-1 = random)", value=-1, step=1)

    st.divider()

    generate_clicked = st.button(
        "Generate Outfit",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.get("initialized", False)
    )


left, right = st.columns([1.1, 1.6], gap="large")


with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Upload Item")

    uploaded_img = st.file_uploader(
        "Upload a fashion item",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img:
        preview_img = ImageOps.exif_transpose(
            Image.open(uploaded_img)
        ).convert("RGB")
        st.image(preview_img, width=300)

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recommended Outfits")

    if not st.session_state.get("initialized", False):
        st.warning(" まずサイドバーでモデルを初期化してください。")
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    st.markdown("</div>", unsafe_allow_html=True)

def explain_score(score: float) -> str:
    if score >= 0.9:
        return "Excellent harmony in silhouette, color, and balance."
    elif score >= 0.8:
        return "Strong stylistic consistency with the item."
    else:
        return "A versatile and well-balanced coordination."

@st.cache_data
def load_image_from_uploader(uploaded_file):
    return Image.open(uploaded_file)

def recommend_with_progress(item_image: Image.Image):
    init_img = ImageOps.exif_transpose(item_image).convert("RGB")
    if resize_to_512:
        init_img = init_img.resize((512, 512))

    all_generated = []
    current_prompt = st.session_state.get('prompt_text', prompt_text)
    current_total_images = st.session_state.get('total_images', total_images)
    current_batch_size = st.session_state.get('batch_size', batch_size)
    current_seed = st.session_state.get('seed', seed)
    
    effective_seed = None if current_seed < 0 else int(current_seed)
    
    progress_text.markdown(f"**Generating {current_total_images} candidates...**")
    progress_bar.progress(0)

    try:
        all_generated = generate_images(
            init_image=init_img,
            prompt=current_prompt,
            total_images=current_total_images,
            batch_size=current_batch_size,
            seed=effective_seed,
        )
    except RuntimeError as e:
        if "Pipeline not initialized" in str(e):
             st.error("Pipeline not initialized. Please click 'Initialize Models' first.")
             return []
        else:
             st.error(f"Image generation failed: {e}")
             return []
    except Exception as e:
        st.error(f"An unexpected error occurred during generation: {e}")
        return []
        
    progress_bar.progress(70) 
    

    progress_text.markdown("**Ranking best matches...**")
    try:
        scored = [(img, score_image_pil(img)) for img in all_generated]
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        return []
        
    scored.sort(key=lambda x: x[1], reverse=True)
    progress_bar.progress(100) 

    return scored[:3]


if generate_clicked:
    st.session_state['prompt_text'] = prompt_text
    st.session_state['total_images'] = total_images
    st.session_state['batch_size'] = batch_size
    st.session_state['seed'] = seed
    
    if not uploaded_img:
        st.warning("Please upload an image.")
    elif not st.session_state.get("initialized", False):
        st.warning("Please initialize models first.")
    else:
        item_image = load_image_from_uploader(uploaded_img)
        
        with st.spinner("Creating outfits…"):
            top3 = recommend_with_progress(item_image)

        if top3:
            with right:
                progress_text.empty()
                progress_bar.empty()
                
                st.markdown("### Top 3 Recommendations")
                
                cols = st.columns(3)
                for i, (img, score) in enumerate(top3):
                    with cols[i]:
                        st.image(img, use_container_width=True)
                        st.caption(f"Match score: **{score:.3f}**")
                        st.markdown(
                            f"<small>{explain_score(score)}</small>",
                            unsafe_allow_html=True,
                        )

                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, (img, _) in enumerate(top3):
                        imgb = io.BytesIO()
                        img.save(imgb, format="PNG")
                        zf.writestr(f"outfit_{i+1}.png", imgb.getvalue())

                st.download_button(
                    "Download Top-3",
                    buf.getvalue(),
                    "outfits_top3.zip",
                    "application/zip",
                )
        else:
             # 生成に失敗した場合
             progress_text.empty()
             progress_bar.empty()
             st.warning("Outfit generation failed. Check error messages above.")
