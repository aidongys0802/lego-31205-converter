import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 资源调度版", layout="wide")
# 1. 零件清单
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
def find_best_lego(target_rgb, current_inv, allowed_colors=None):
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in current_inv.items():
       if count <= 0: continue
       if allowed_colors and name not in allowed_colors: continue
       # 颜色感知加权
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist, best_name = dist, name
   current_inv[best_name][1] -= 1
   return current_inv[best_name][0], best_name
# --- 侧边栏 ---
st.sidebar.header("🎨 皮肤与库存优化")
face_dither_prob = st.sidebar.slider("面部高光(White)抖动密度", 0.0, 0.5, 0.15, help="在皮肤中随机混入白色的概率，用以节省肤色积木")
brightness = st.sidebar.slider("图像亮度", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("五官锐度", 0.5, 2.5, 1.4)
zoom = st.sidebar.slider("人脸缩放", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 预处理
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   # 检测与裁剪
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       dim = int(min(w, h) / zoom)
       img_cropped = img_pil.crop((max(0, cx-dim//2), max(0, cy-dim//2), min(w, cx+dim//2), min(h, cy+dim//2)))
   else:
       dim = min(w, h)
       img_cropped = img_pil.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   # 缩小至像素网格
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small).astype(float)
   # 获取面部 ROI
   small_cv = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((grid_res, grid_res), dtype=bool)
   for (sx, sy, sw, sh) in small_faces:
       face_mask[sy:sy+sh, sx:sx+sw] = True
   # 运行库存
   run_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_INVENTORY.items()}
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 重点：定义两套受限色板
   # 皮肤色板：包含 White，用于稀释 Flesh 压力
   skin_palette = ["Flesh", "Tan", "White", "Dark Orange", "Reddish Brown", "Black", "Red"]
   # 背景色板：严格禁止使用 White，由浅蓝/浅绿/灰色代替
   bg_palette = [k for k in LEGO_INVENTORY.keys() if k not in ["White", "Flesh", "Tan"]]
   # 备选背景色（如果上面的用完了，才允许用极少量的 Tan）
   bg_palette_extended = bg_palette + ["Medium Stone Grey", "Light Aqua", "Sand Blue"]
   # 渲染逻辑
   for y in range(grid_res):
       for x in range(grid_res):
           target = pixel_array[y, x]
           if face_mask[y, x]:
               # 面部逻辑：引入随机白色干扰，实现“稀疏抖动”
               if random.random() < face_dither_prob:
                   # 强行寻找最接近白色的肤色表现
                   rgb, _ = find_best_lego(target, run_inv, allowed_colors=["White", "Flesh"])
               else:
                   rgb, _ = find_best_lego(target, run_inv, allowed_colors=skin_palette)
           else:
               # 背景逻辑：禁止白/肉色，保护核心库存
               rgb, _ = find_best_lego(target, run_inv, allowed_colors=bg_palette_extended)
           canvas[y, x] = rgb
   # 展示
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), caption="优化调度预览", use_container_width=True)
   with st.expander("📊 零件安全余量监控"):
       st.table([{"零件": k, "消耗": LEGO_INVENTORY[k][1]-v[1], "库内剩余": v[1]} for k, v in run_inv.items()])
