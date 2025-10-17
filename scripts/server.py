#!/usr/bin/env python3
# server.py
import os
from typing import Optional, List
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------- 模型封装 ----------------
class QwenVLWrapper:
    def __init__(self, model_dir: str):
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def convert_to_user_message(self, prompt: str, image_path: Optional[str] = None):
        content = [{"type": "text", "text": f" {prompt}"}]
        if image_path:
            content.append({"type": "image", "image": image_path})
        return {"role": "user", "content": content}

    @torch.inference_mode()
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        print(prompt)
        messages: List[dict] = [self.convert_to_user_message(prompt, image_path)]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        gen_ids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        return self.processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

# ---------------- FastAPI ----------------
class InferenceRequest(BaseModel):
    prompt: str
    image_path: Optional[str] = None  # 兼容旧方式：服务器本地路径

class InferenceResponse(BaseModel):
    result: str

app = FastAPI(title="Qwen-VL Inference Server")
model_dir = "model/vla_r1"
qwen = QwenVLWrapper(model_dir)

@app.get("/health")
def health():
    return {"ok": True}

# 旧：仍然支持传服务器本地路径（可留可删）
@app.post("/generate", response_model=InferenceResponse)
def generate(req: InferenceRequest):
    if req.image_path and not os.path.isfile(req.image_path):
        raise HTTPException(status_code=400, detail="Image path does not exist.")
    try:
        result = qwen.generate(req.prompt, req.image_path)
        return InferenceResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 新：直接上传图片文件（推荐用这个）
@app.post("/generate_multipart", response_model=InferenceResponse)
async def generate_multipart(
    prompt: str = Form(...),
    image: UploadFile = File(None)  # 可选；仅文本也能跑
):
    tmp_path = None
    try:
        if image is not None:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            tmp_path = f"/tmp/{uuid4().hex}_{image.filename}"
            with open(tmp_path, "wb") as f:
                f.write(await image.read())

        out = qwen.generate(prompt, tmp_path)
        return InferenceResponse(result=out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    # 生产建议监听 127.0.0.1 后用隧道/Nginx暴露；调试可 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
