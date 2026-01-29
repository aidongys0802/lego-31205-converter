import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 稳定版", layout="wide")
# 1. 定义积木库
LEGO_DATA = {
   "Black": (0, 0, 0), "Dark Stone Grey": (99, 95, 97),
   "Medium Stone Grey": (150, 152, 152), "White": (255, 255, 255),
   "Navy Blue": (27, 42, 52), "Blue": (0, 85, 191),
   "Medium Azure": (0, 174, 216), "Light Aqua": (188, 225, 233),
   "Tan": (217, 187, 123), "Flesh": (255, 158, 146),
   "Dark Orange": (168, 84, 9), "Reddish Brown": (127, 51, 26),
   "Red": (215, 0, 0), "Medium Lavender": (156, 124, 204),
   "Sand Blue": (112, 129, 154)
}
# 2. 高可靠性颜色匹配
def get_lego_color(target_rgb, is_skin=False, dither_rate=0.0):
   # 溢出保护
   tr, tg, tb = [int(np.clip(c, 0, 255)) for c in target_rgb]
   # 皮肤微量随机高光
   if is_skin and dither_rate > 0 and random.random() < dither_rate:
       tr, tg, tb = [min(255, c + 30) for c in [tr, tg, tb]]
   best_dist = float('inf')
   best_rgb = (0, 0, 0)
   for name, rgb in LEGO_DATA.items():
       # 肤色区域限定：只使用暖色系
       if is_skin and name not in ["Flesh", "Tan", "White", "Dark Orange", "Reddish Brown"]:
           continue
       # 采用感知权重距离
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist = dist
           best_rgb = rgb
   return best_rgb
# 3. UI 布局
st.sidebar.header("⚙️ 核心参数")
grid_res = st.sidebar.select_slider("网格精度", options=[32, 48, 64], value=48)
zoom = st.sidebar.slider("人脸对焦", 1.0, 3.0, 1.8)
contrast = st.sidebar.slider("五官清晰度", 1.0, 2.0, 1.3)
skin_dither = st.sidebar.slider("皮肤质感点", 0.0, 0.3, 0.05)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 预处理：加载并统一尺寸
   img_pil = Image.open(uploaded_file).convert("RGB")
   # 简单的对比度增强
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   # 正方形裁剪逻辑 (确保不乱码的基础)
   w, h = img_pil.size
   crop_size = int(min(w, h) / zoom)
   left = (w - crop_size) // 2
   top = (h - crop_size) // 2
   img_cropped = img_pil.crop((left, top, left + crop_size, top + crop_size))
   # 缩放到目标乐高网格
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small).copy()
   # 肤色检测 (HSV 空间更准)
   hsv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2HSV)
   # 典型的肤色范围
   skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
   # 创建画布 (高度, 宽度, 颜色通道)
   # 这一步是防止乱码的关键：显式初始化
   output_array = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 逐像素点精准填充
   for y in range(grid_res):
       for x in range(grid_res):
           is_skin = skin_mask[y, x] > 0
           # 这里的坐标映射必须严格一致
           lego_rgb = get_lego_color(pixel_array[y, x], is_skin=is_skin, dither_rate=skin_dither)
           output_array[y, x] = lego_rgb
   # 显示结果
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="原始裁剪预览", use_container_width=True)
   with col2:
       res_img = Image.fromarray(output_array)
       # 放大显示时使用 NEAREST 保持像素感
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST),
                caption="乐高转换预览", use_container_width=True)
