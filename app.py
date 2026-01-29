import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 皮肤优先版", layout="wide")
# 零件库原始数据
LEGO_INVENTORY = {
   "Black": [(0, 0, 0), 600], "Dark Stone Grey": [(99, 95, 97), 470],
   "Medium Stone Grey": [(150, 152, 152), 370], "White": [(255, 255, 255), 350],
   "Navy Blue": [(27, 42, 52), 310], "Blue": [(0, 85, 191), 280],
   "Medium Azure": [(0, 174, 216), 170], "Light Aqua": [(188, 225, 233), 140],
   "Tan": [(217, 187, 123), 140], "Flesh": [(255, 158, 146), 140],
   "Dark Orange": [(168, 84, 9), 110], "Reddish Brown": [(127, 51, 26), 100],
   "Red": [(215, 0, 0), 100], "Medium Lavender": [(156, 124, 204), 100],
   "Sand Blue": [(112, 129, 154), 100]
}
@st.cache_resource
def load_cascade():
   return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_best_color(target_rgb, current_inv, force_palette=None, dither_rate=0.0):
   tr, tg, tb = [int(np.clip(c, 0, 255)) for c in target_rgb]
   # 肤色区域的稀疏高光处理
   if force_palette and dither_rate > 0 and random.random() < dither_rate:
       tr, tg, tb = [min(255, c + 35) for c in [tr, tg, tb]]
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in current_inv.items():
       if count <= 0: continue
       # 如果指定了色板（如皮肤优先色板），则只从这些颜色里选
       if force_palette and name not in force_palette: continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist, best_name = dist, name
   current_inv[best_name][1] -= 1
   return current_inv[best_name][0], best_name
# --- 侧边栏 ---
st.sidebar.header("🎛️ 图像控制")
brightness = st.sidebar.slider("1. 整体亮度", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("2. 五官锐度", 0.5, 2.5, 1.4)
skin_dither = st.sidebar.slider("3. 皮肤稀疏抖动", 0.0, 0.4, 0.05)
zoom = st.sidebar.slider("4. 对焦范围", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48
   # 裁剪
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       crop_dim = int(min(w, h) / zoom)
       left, top = max(0, cx - crop_dim//2), max(0, cy - crop_dim//2)
       img_cropped = img_pil.crop((left, top, min(w, left+crop_dim), min(h, top+crop_dim)))
   else:
       dim = min(w, h)
       img_cropped = img_pil.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small)
   # 检测面部 Mask
   small_cv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((grid_res, grid_res), dtype=bool)
   if len(small_faces) > 0:
       for (sx, sy, sw, sh) in small_faces:
           face_mask[sy:sy+sh, sx:sx+sw] = True
   # 初始化
   run_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_INVENTORY.items()}
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   filled = np.zeros((grid_res, grid_res), dtype=bool)
   # --- 第一步：优先填补面部区域 (VIP 通道) ---
   # 定义面部允许的色板（含妆造识别）
   skin_palette = ["Flesh", "White", "Tan", "Dark Orange", "Reddish Brown", "Black",
                   "Red", "Medium Lavender", "Medium Azure", "Blue"]
   for y in range(grid_res):
       for x in range(grid_res):
           if face_mask[y, x]:
               rgb, _ = get_best_color(pixel_array[y, x], run_inv,
                                       force_palette=skin_palette,
                                       dither_rate=skin_dither)
               canvas[y, x] = rgb
               filled[y, x] = True
   # --- 第二步：填补剩余区域 (衣服、帽子、头发、背景) ---
   for y in range(grid_res):
       for x in range(grid_res):
           if not filled[y, x]:
               # 这里不传入 force_palette，意味着可以使用所有剩余零件
               rgb, _ = get_best_color(pixel_array[y, x], run_inv)
               canvas[y, x] = rgb
   # 展示结果
   col1, col2 = st.columns(2)
   with col1: st.image(img_cropped, use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), use_container_width=True)
   with st.expander("📊 零件消耗统计"):
       st.table([{"颜色": k, "已用": LEGO_INVENTORY[k][1]-v[1]} for k, v in run_inv.items()])
