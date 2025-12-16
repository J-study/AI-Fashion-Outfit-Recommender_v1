from typing import List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cpu"
pipe = None
clf_model = None
clf_tf = None

def init_sd(pipe_path: Optional[str] = None, lora_repo_id: Optional[str] = None, lora_weight_name: Optional[str] = None):
    global pipe
    model_id = pipe_path if pipe_path else "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    if lora_repo_id:
        pipe.load_lora_weights(lora_repo_id, weight_name=lora_weight_name if lora_weight_name else None)
    return pipe

def generate_images(
    init_image: Image.Image,
    prompt: str,
    total_images: int = 10,
    batch_size: int = 2,
    strength: float = 0.9,
    guidance_scale: float = 9.0,
    num_inference_steps: int = 30,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[Image.Image]:
    global pipe
    if pipe is None:
        raise RuntimeError("Pipeline not initialized.")
    if init_image is None:
        raise ValueError("init_image is required.")

    if negative_prompt is None:
        negative_prompt = "anime, illustration, low quality, cut off, cropped, nsfw, blurry, bad hands, bad anatomy"

    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(int(seed))

    img_w, img_h = init_image.size
    out_w = width if width is not None else img_w
    out_h = height if height is not None else img_h

    images_out: List[Image.Image] = []
    num_batches = (total_images + batch_size - 1) // batch_size

    for _ in range(num_batches):
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            width=out_w,
            height=out_h,
            generator=generator,
        )
        images_out.extend(result.images)

    return images_out[:total_images]

def init_classifier(model_obj, transform):
    global clf_model, clf_tf
    clf_model = model_obj
    clf_model.eval()
    clf_tf = transform

def score_image_pil(img: Image.Image) -> float:
    if clf_model is None or clf_tf is None:
        raise ValueError("Classifier not initialized.")
    x = clf_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf_model(x)
        prob = F.softmax(logits, dim=1)[0]
    return float(prob[1].cpu().item())
