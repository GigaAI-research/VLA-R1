# -*- coding: utf-8 -*-
import io
import re
import json
import math
import requests
import numpy as np
from PIL import Image

SERVER_URL = "http://127.0.0.1:8000"
aff_form=" Please output the 2D affordance region define by bounding box of the object in the format <output>{\"x\": value, \"y\": value, \"width\": value, \"height\": value}</output>.\n<image>"
traj_form=" Please output the optimal trajectory of the end-effector to the target object in the format <output>[[x1, y1], [x2, y2], ...]</output>.\n<image>"    

def encode_image_to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    """
    img: np.ndarray, HxW 或 HxWxC，dtype 可以是 uint8/float
    返回: JPEG 二进制 bytes
    """
    if img.ndim == 2:  
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        im =Image.fromarray(arr, mode="L")
    else:

        arr = img[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)  
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()

def parse_output_block(text):
    """
    在整段千问回复中抓取 <output> ... </output> 并转成 Python 对象
    """
    m = re.search(r"<output>(.*?)</output>", text, flags=re.S)
    if m is None:
        raise ValueError("No <output>...</output> block found in model reply")

    raw = m.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

def ask_server(img_rgb_np: np.ndarray, prompt):
    server_url=SERVER_URL
    url = server_url.rstrip("/") + "/generate_multipart"
    jpg_bytes = encode_image_to_jpeg_bytes(img_rgb_np)
    files = {"image": ("frame.jpg", jpg_bytes, "image/jpeg")}

    data  = {
        "prompt": prompt
    }
    r = requests.post(url, data=data, files=files, timeout=600)
    r.raise_for_status()
    text = r.json().get("result", "")
    return text


if __name__ == "__main__":
    #affordance
    img = Image.open("data/sample/affrodance.png").convert("RGB")
    img = np.array(img)
    prompt="pick up the broccoli and place it in a pot."
    tk=ask_server(img,prompt+aff_form)
    print(tk)
    answer=parse_output_block(tk)
    print(answer)
    #trajectory
    img = Image.open("data/sample/trajectory.png").convert("RGB")
    img = np.array(img)
    prompt="reach for the ketchup bottle on the oven."
    tk=ask_server(img,prompt+traj_form)
    print(tk)
    answer=parse_output_block(tk)
    print(answer)
