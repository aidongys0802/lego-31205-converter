import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 终极全控版", layout="wide")
# 1. 严格库存数据
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
def get_best_color(target_rgb, current_inv, is_face_roi=False, dither_rate=0.0):
   tr, tg, tb = [int(np.clip(c, 0, 255)) for c in target_rgb]
   # 皮肤基础色板
   skin_base = ["Flesh", "White", "Tan", "Dark Orange", "Reddish Brown", "Black"]
   # 妆造/彩色增强色板
   makeup_colors = ["Red", "Medium Lavender", "Medium Azure", "Blue", "Light Aqua", "Navy Blue"]
   # 皮肤区域稀疏高光处理
   if is_face_roi and dither_rate > 0 and random.random() < dither_rate:
       tr, tg, tb = [min(255, c + 35) for c in [tr, tg, tb]]
   # 计算饱和度用于妆造检测
   max_c, min_c = max(tr, tg, tb), min(tr, tg, tb)
   sat = (max_c - min_c) / (max_c + 0.1)
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in current_inv.items():
       if count <= 0: continue
       if is_face_roi:
           # 逻辑：如果饱和度低用肤色，饱和度高允许妆造色，五官核心始终允许黑色
           if sat < 0.25 and name not in skin_base: continue
           if sat >= 0.25 and name not in (skin_base + makeup_colors): continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist, best_name = dist, name
   current_inv[best_name][1] -= 1
   return current_inv[best_name][0], best_name
# --- 侧边栏所有滑块回归 ---
st.sidebar.header("🎛️ 图像增强 (全保留)")
brightness = st.sidebar.slider("1. 整体亮度 (肤色基调)", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("2. 五官锐度 (对比度)", 0.5, 2.5, 1.4)
skin_dither = st.sidebar.slider("3. 皮肤质感点 (稀疏抖动)", 0.0, 0.4, 0.05)
st.sidebar.header("📐 构图控制")
zoom = st.sidebar.slider("对焦范围 (Zoom)", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # A. 预处理
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   # B. 人脸与裁剪
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       crop_dim = int(min(w, h) / zoom)
       left, top = max(0, cx - crop_dim//2), max(0, cy - crop_dim//2)
       img_cropped = img_pil.crop((left, top, min(w, left+crop_dim), min(h, top+crop_dim)))
   else:
       dim = min(w, h)
       img_cropped = img_pil.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   # C. 渲染逻辑
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small)
   # 获取面部 ROI
   small_cv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((grid_res, grid_res), dtype=bool)
   if len(small_faces) > 0:
       for (sx, sy, sw, sh) in small_faces:
           face_mask[sy:sy+sh, sx:sx+sw] = True
   run_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_INVENTORY.items()}
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 渲染每一个像素
   for y in range(grid_res):
       for x in range(grid_res):
           is_face = face_mask[y, x]
           rgb, _ = get_best_color(pixel_array[y, x], run_inv, is_face_roi=is_face, dither_rate=skin_dither)
           canvas[y, x] = rgb
   # D. 展示
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="图像预处理预览", use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), caption="乐高艺术转换", use_container_width=True)
   # E. 消耗清单
   with st.expander("📊 查看详细零件消耗"):
       stats = []
       for name, original in LEGO_INVENTORY.items():
           used = original[1] - run_inv[name][1]
           stats.append({"颜色": name, "已用": used, "剩余": run_inv[name][1]})
       st.table(stats)
