import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 艺术妆造版", layout="wide")
# 1. 严格零件清单
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
def get_best_color(target_rgb, current_inv, is_face_roi=False):
   tr, tg, tb = [int(c) for c in target_rgb]
   # 肤色区域限定色板
   skin_base = ["Flesh", "White", "Tan", "Dark Orange", "Reddish Brown", "Black"]
   # 扩展艺术色板（妆造用）
   makeup_colors = ["Red", "Medium Azure", "Blue", "Medium Lavender", "Light Aqua"]
   # 计算饱和度，判断是否是彩色妆造
   max_c = max(tr, tg, tb)
   min_c = min(tr, tg, tb)
   saturation = (max_c - min_c) / (max_c + 0.1)
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in current_inv.items():
       if count <= 0: continue
       # 逻辑：在人脸区域内
       if is_face_roi:
           # 如果饱和度低，强制用皮肤色板；如果饱和度高且在妆造色板中，允许使用
           if saturation < 0.25 and name not in skin_base:
               continue
           if saturation >= 0.25 and name not in (skin_base + makeup_colors):
               continue
       # 感知距离计算
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist = dist
           best_name = name
   current_inv[best_name][1] -= 1
   return current_inv[best_name][0], best_name
# --- UI 侧边栏 ---
st.sidebar.header("🎨 图像调节")
contrast_val = st.sidebar.slider("对比度 (让五官更锐利)", 0.5, 2.5, 1.4)
brightness_val = st.sidebar.slider("亮度 (调节皮肤基底)", 0.5, 2.0, 1.1)
st.sidebar.header("📐 构图控制")
zoom_val = st.sidebar.slider("人脸对焦范围", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传人像照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 加载与增强
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_val)
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness_val)
   # 人脸检测
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48
   # 裁剪逻辑
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       crop_dim = int(min(w, h) / zoom_val)
       left, top = max(0, cx - crop_dim//2), max(0, cy - crop_dim//2)
       img_cropped = img_pil.crop((left, top, min(w, left+crop_dim), min(h, top+crop_dim)))
   else:
       dim = min(w, h)
       img_cropped = img_pil.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   # 缩放至乐高比例
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small)
   # 获取面部 ROI Mask
   small_cv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_roi = np.zeros((grid_res, grid_res), dtype=bool)
   if len(small_faces) > 0:
       for (sx, sy, sw, sh) in small_faces:
           face_roi[sy:sy+sh, sx:sx+sw] = True
   # 运行分配逻辑
   run_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_INVENTORY.items()}
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 渲染
   for y in range(grid_res):
       for x in range(grid_res):
           is_face = face_roi[y, x]
           rgb, _ = get_best_color(pixel_array[y, x], run_inv, is_face_roi=is_face)
           canvas[y, x] = rgb
   # 展示
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="图像预处理预览", use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST),
                caption="乐高 48x48 艺术转换", use_container_width=True)
   # 零件清单
   with st.expander("📊 查看详细零件消耗"):
       st.table([{"零件颜色": k, "已使用": 600-v[1] if k=="Black" else LEGO_INVENTORY[k][1]-v[1]} for k, v in run_inv.items()])
