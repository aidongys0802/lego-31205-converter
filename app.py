import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 修正版", layout="wide")
# 1. 严格零件清单
def get_fresh_inventory():
   return {
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
# 2. 改进的匹配算法
def match_color_v2(target_rgb, inv, is_face, dither_strength):
   tr, tg, tb = target_rgb
   # 在人脸区域，小概率随机微调亮度，产生 Tan/White 交错效果
   if is_face and random.random() < dither_strength:
       shift = random.randint(15, 35)
       tr, tg, tb = min(255, tr+shift), min(255, tg+shift), min(255, tb+shift)
   best_dist = float('inf')
   best_name = None
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       # 核心逻辑：如果在面部，给肤色零件极大权重
       if is_face:
           if name in ["Flesh", "Tan", "White"]: dist *= 0.5
       else:
           # 如果不在面部，大幅增加使用肤色零件的“代价”，强迫背景选灰色/蓝色
           if name in ["Flesh", "Tan", "White"]: dist *= 5.0
       if dist < best_dist:
           best_dist = dist
           best_name = name
   if best_name:
       inv[best_name][1] -= 1 # 实时扣减
       return inv[best_name][0], best_name
   return (0,0,0), "Black"
# --- UI ---
st.sidebar.header("🎨 调节面板")
brightness = st.sidebar.slider("亮度", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("锐度/对比度", 0.5, 2.5, 1.4)
dither = st.sidebar.slider("面部混色颗粒度", 0.0, 0.5, 0.2)
zoom = st.sidebar.slider("对焦", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   raw_inv = get_fresh_inventory()
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   # 裁剪与 ROI
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       dim = int(min(w, h) / zoom)
       img_cropped = img_pil.crop((max(0,cx-dim//2), max(0,cy-dim//2), min(w,cx+dim//2), min(h,cy+dim//2)))
   else:
       img_cropped = img_pil.crop(((w-min(w,h))//2, (h-min(w,h))//2, (w+min(w,h))//2, (h+min(w,h))//2))
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small)
   # 重新在小图找脸确定 ROI
   small_cv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((grid_res, grid_res), dtype=bool)
   for (sx, sy, sw, sh) in small_faces:
       face_mask[sy:sy+sh, sx:sx+sw] = True
   # 渲染
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 这一步极其重要：先复制一份库存用于当前计算
   current_inv = {k: [list(v[0]), v[1]] for k, v in raw_inv.items()}
   for y in range(grid_res):
       for x in range(grid_res):
           rgb, _ = match_color_v2(pixel_array[y, x], current_inv, face_mask[y, x], dither)
           canvas[y, x] = rgb
   col1, col2 = st.columns(2)
   with col1: st.image(img_cropped, use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), use_container_width=True)
   # 统计表
   with st.expander("📊 零件消耗统计（严格核对）"):
       summary = []
       original_inv = get_fresh_inventory()
       for k in original_inv.keys():
           used = original_inv[k][1] - current_inv[k][1]
           summary.append({"颜色": k, "已使用": used, "剩余": current_inv[k][1]})
       st.table(summary)
