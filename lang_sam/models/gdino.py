import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from lang_sam.models.utils import get_device_type

device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self):
        self.build_model()

    def build_model(self, ckpt_path: str | None = None):
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            DEVICE
        )

    def predict(
        self,
        pil_images: list[Image.Image],
        text_prompt: list[str],
        threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        
        if isinstance(text_prompt, list):
            for i, prompt in enumerate(text_prompt):
                if not prompt.endswith("."):
                    text_prompt[i] = prompt + "."
        elif isinstance(text_prompt, str):
            if not text_prompt.endswith("."):
                text_prompt += "."
        else:
            raise TypeError("text_prompt must be either a list or a string")

        inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        return results


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
        ["wheel", "wheel"],
        0.3,
        0.25,
    )
    print(out)
