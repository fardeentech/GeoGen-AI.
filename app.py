"""Streamlit interface for generating tiles from a trained conditional GAN."""
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
from torchvision.utils import make_grid, save_image

from models import CLASS_NAMES, Generator

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "conditional_dcgan_latest.pt"


def image_to_png(image: torch.Tensor) -> bytes:
    """Encode one display-range CHW RGB tensor as a PNG download."""
    output = BytesIO()
    save_image(image.detach().cpu().clamp(0, 1), output, format="PNG")
    return output.getvalue()


@st.cache_resource
def load_generator(path: str, mtime: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator = Generator(latent_dim=checkpoint["latent_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, checkpoint, device


def main() -> None:
    st.set_page_config(page_title="GeoGen AI", page_icon="🛰️", layout="wide")
    st.title("🛰️ GeoGen AI")
    st.subheader("Satellite image generation and data augmentation using GANs")
    st.write("Choose a EuroSAT land-use class and generate new 64 × 64 RGB tiles using the trained conditional GAN.")

    if not CHECKPOINT.exists():
        st.warning("No trained generator checkpoint found. Train the model in Google Colab, then copy `conditional_dcgan_latest.pt` into `artifacts/checkpoints/`.")
        st.code("python train_gan.py --epochs 50", language="bash")
        st.stop()

    generator, checkpoint, device = load_generator(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
    class_names = checkpoint.get("class_names") or CLASS_NAMES
    with st.sidebar:
        st.header("Generation settings")
        selected = st.selectbox("Land-use class", class_names)
        count = st.slider("Number of images", min_value=1, max_value=16, value=4)
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)

    if st.button("Generate satellite tiles", type="primary"):
        torch.manual_seed(int(seed))
        class_id = class_names.index(selected)
        with torch.no_grad():
            labels = torch.full((count,), class_id, device=device, dtype=torch.long)
            images = generator(torch.randn(count, checkpoint["latent_dim"], device=device), labels)
        display = ((images.cpu() + 1) / 2).clamp(0, 1)
        grid = make_grid(display, nrow=min(4, count), padding=8)
        st.image(grid.permute(1, 2, 0).numpy(), caption=f"Generated class: {selected}", use_container_width=True)
        st.download_button("Download the first generated image (.png)", data=image_to_png(display[0]), file_name=f"geogen_{selected}_{seed}.png", mime="image/png")

    st.caption("Generated tiles are model outputs, not real satellite observations. They should be visually and scientifically validated before research use.")


if __name__ == "__main__":
    main()

