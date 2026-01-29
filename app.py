import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 艺术大师修复版", layout="wide")
# 1. 零件库数据
def get_lego_inventory():
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
# 2. 增强型颜色匹配逻辑
def find_color_logic(target_rgb, inv, is_skin=False, sparse_dither=0.0):
   tr, tg, tb = target_rgb
   # 皮肤随机微量高光 (抖动处理)
   if is_skin and sparse_dither > 0 and random.random() < sparse_dither:
       # 向白色方向偏移，但由于 clip 保护，不会报错
       tr, tg, tb = tr + 40, tg + 40, tb + 40
   best_dist = float('inf')
   best_key = "Black"
   # 将输入限制在合法范围
   tr, tg, tb = np.clip([tr, tg, tb], 0, 255)
   for name, data in inv.items():
       (r, g, b), count = data
       if count > 0:
           # 皮肤限制色板
           if is_skin and name not in ["Flesh", "Tan", "White", "Dark Orange", "Reddish Brown"]:
               continue
           # 权重距离计算
           dr, dg, db = r - tr, g - tg, b - tb
           dist = 2*dr**2 + 4*dg**2 + 3*db**2
           if dist < best_dist:
               best_dist, best_key = dist, name
   inv[best_key][1] -= 1
   return inv[best_key][0]
# 3. UI 交互
st.sidebar.header("🛠️ 模式设置")
performance_mode = st.sidebar.toggle("性能模式 (预览快/画质低)", value=False)
st.sidebar.header("🎨 艺术控制")
skin_dither_val = st.sidebar.slider("皮肤抖动点密度", 0.0, 0.4, 0.05)
contrast = st.sidebar.slider("对比度 (勾勒五官)", 1.0, 2.5, 1.4)
brightness = st.sidebar.slider("整体亮度", 0.5, 1.5, 1.1)
grid_size = st.sidebar.select_slider("分辨率", options=[32, 48, 64], value=48)
zoom = st.sidebar.slider("缩放对焦", 1.0, 3.0, 2.0)
uploaded_file = st.file_uploader("上传人像", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # --- 阶段 A: 图片预处理 ---
   raw_img = Image.open(uploaded_file).convert("RGB")
   # 图像增强 (这里要保护溢出)
   enhancer_c = ImageEnhance.Contrast(raw_img)
   img_c = enhancer_c.enhance(contrast)
   enhancer_b = ImageEnhance.Brightness(img_c)
   img_p = enhancer_b.enhance(brightness)
   # 智能裁剪
   cv_img = cv2.cvtColor(np.array(img_p), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_p.size
   if len(faces) > 0:
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       dim = int(min(w, h) / zoom)
       img_cropped = img_p.crop((max(0, cx-dim//2), max(0, cy-dim//2), min(w, cx+dim//2), min(h, cy+dim//2)))
   else:
       dim = min(w, h)
       img_cropped = img_p.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="对焦及光影预览", use_container_width=True)
   # --- 阶段 B: 核心计算 ---
   if not performance_mode:
       with st.spinner("深度计算颜色分配 (避开乱码)..."):
           # 缩放至像素格
           small = img_cropped.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
           pixels = np.array(small, dtype=np.int16) # 用 int16 防止计算中溢出
           # 肤色识别 (HSV)
           hsv_small = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2HSV)
           lower_skin = np.array([0, 25, 70])
           upper_skin = np.array([25, 255, 255])
           skin_mask = cv2.inRange(hsv_small, lower_skin, upper_skin)
           # 初始化结果
           inv = {k: [list(v[0]), v[1]] for k, v in get_lego_inventory().items()}
           canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
           # 按像素点循环填充，确保不重不漏
           for y in range(grid_size):
               for x in range(grid_size):
                   is_skin_pixel = skin_mask[y, x] > 0
                   rgb_val = find_color_logic(
                       pixels[y, x],
                       inv,
                       is_skin=is_skin_pixel,
                       sparse_dither=skin_dither_val
                   )
                   canvas[y, x] = rgb_val
           with col2:
               final_preview = Image.fromarray(canvas)
               st.image(final_preview.resize((600, 600), Image.Resampling.NEAREST),
                        caption="乐高艺术输出", use_container_width=True)
   else:
       # 极速模式预览
       fast_small = img_cropped.resize((grid_size, grid_size), Image.Resampling.NEAREST)
       with col2:
           st.image(fast_small.resize((600, 600), Image.Resampling.NEAREST), caption="性能模式快速预览", use_container_width=True)
