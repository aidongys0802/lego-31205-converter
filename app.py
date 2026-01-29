import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
st.set_page_config(page_title="LEGO 31205 艺术版", layout="wide")
# 1. 零件清单
def get_inventory():
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
# 2. 4x4 拜耳矩阵（用于产生整齐的颗粒感）
BAYER_4X4 = np.array([
   [ 0,  8,  2, 10],
   [12,  4, 14,  6],
   [ 3, 11,  1,  9],
   [15,  7, 13,  5]
]) / 16.0
def match_lego_color(tr, tg, tb, inv, is_skin, dither_val):
   # 应用结构化抖动偏移
   tr, tg, tb = tr + dither_val, tg + dither_val, tb + dither_val
   tr, tg, tb = np.clip([tr, tg, tb], 0, 255)
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       # 权重控制：面部优先 Flesh/Tan/White
       if is_skin:
           if name in ["Flesh", "Tan", "White"]: dist *= 0.5
       else:
           # 背景绝对禁止抢 Flesh/Tan，白色也提高代价
           if name in ["Flesh", "Tan"]: dist *= 20.0
           if name == "White": dist *= 3.0
       if dist < best_dist:
           best_dist, best_name = dist, name
   inv[best_name][1] -= 1
   return inv[best_name][0]
# --- UI ---
st.sidebar.header("🎨 核心控制")
brightness = st.sidebar.slider("亮度 (肤色基调)", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("对比度 (五官清晰度)", 0.5, 2.5, 1.4)
dither_scale = st.sidebar.slider("颗粒抖动强度", 0, 60, 30)
zoom = st.sidebar.slider("人脸缩放", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传人像照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 预处理
   img = Image.open(uploaded_file).convert("RGB")
   img = ImageEnhance.Brightness(img).enhance(brightness)
   img = ImageEnhance.Contrast(img).enhance(contrast)
   # 裁剪与缩放
   w, h = img.size
   side = int(min(w, h) / zoom)
   left, top = (w - side) // 2, (h - side) // 2
   img_cropped = img.crop((left, top, left + side, top + side))
   small = img_cropped.resize((48, 48), Image.Resampling.LANCZOS)
   pixel_data = np.array(small, dtype=float)
   # 肤色区域识别 (HSV 空间更准确)
   hsv = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2HSV)
   # 典型的肤色范围
   skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255])) > 0
   # 渲染
   current_inv = {k: [list(v[0]), v[1]] for k, v in get_inventory().items()}
   canvas = np.zeros((48, 48, 3), dtype=np.uint8)
   for y in range(48):
       for x in range(48):
           # 获取当前位置的抖动偏移量
           bayer_val = BAYER_4X4[y % 4, x % 4] - 0.5
           dither_offset = bayer_val * dither_scale
           rgb = match_lego_color(
               pixel_data[y, x, 0], pixel_data[y, x, 1], pixel_data[y, x, 2],
               current_inv, skin_mask[y, x], dither_offset
           )
           canvas[y, x] = rgb
   # 显示结果
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), caption="颗粒感抖动效果", use_container_width=True)
   # 统计
   with st.expander("📊 零件消耗详单"):
       raw = get_inventory()
       st.table([{"颜色": k, "已用": raw[k][1]-v[1], "剩余": v[1]} for k, v in current_inv.items()])
