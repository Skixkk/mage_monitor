'''
Author: Skixkk <166358870+Skixkk@users.noreply.github.com>
Date: 2026-10-05 18:32:31
LastEditors: Skixkk <166358870+Skixkk@users.noreply.github.com>
LastEditTime: 2026-10-05 21:02:25
FilePath: /mage_monitor/main.py
Description:  mage 逻辑点测试
'''


from pathlib import Path
import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ---------------------- 全局计时起点：脚本总运行开始 ----------------------
total_start_time = time.time()

MODEL_PATH = Path(r"D:\product\mage_monitor\mage-vl-local")

# ---------------------- 模型加载计时开始 ----------------------
model_load_start = time.time()

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

# 模型加载完成，计算耗时
model_load_cost = time.time() - model_load_start

if __name__ == "__main__":
    print(f"【模型加载(读取权重+迁移cuda)耗时】: {model_load_cost:.2f} 秒")
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

    # ---------------------- 推理计时开始（仅generate生成阶段） ----------------------
    infer_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600
        )
    infer_cost = time.time() - infer_start

    result = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 脚本整体总耗时（从脚本第一行到输出完成）
    total_cost = time.time() - total_start_time

    print(f"【模型推理(generate生成)耗时】: {infer_cost:.2f} 秒")
    print(f"【脚本运行总时长】: {total_cost:.2f} 秒")
    print("==========模型识别结果==========")
    print(result)
