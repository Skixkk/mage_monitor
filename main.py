from pathlib import Path
import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_PATH = Path(r"D:\product\mage_monitor\mage-vl-local")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float16
)
model = model.to("cuda")

if __name__ == "__main__":
    print("Model device:", next(model.parameters()).device)

    img_path = Path("./assets/images/test.jpg")
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((960, 720))  # 进一步降低分辨率，减少图像编码压力

    prompt = "客观描述图像内出现的物体、场景、环境，只输出画面内容"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt"
    ).to("cuda", torch.float16)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600
        )
    cost = time.time() - start

    result = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"推理耗时：{cost:.2f} 秒")
    print("==========模型识别结果==========")
    print(result)
