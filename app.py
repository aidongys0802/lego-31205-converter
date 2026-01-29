import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 艺术版", layout="wide")
# 1. 初始化库存
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
# 2. 颜色匹配逻辑 (带颗粒度控制)
def find_color(target_rgb, inv, is_face, dither_strength):
   tr, tg, tb = target_rgb
   # 在人脸区域制造随机偏移，诱导 Tan 和 White 混色
   if is_face and dither_strength > 0:
       offset = random.uniform(-40, 40) * dither_strength
       tr, tg, tb = tr + offset, tg + offset, tb + offset
   # 安全限值，防止乱码
   tr, tg, tb = np.clip([tr, tg, tb], 0, 255)
   best_dist = float('inf')
   best_key = "Black"
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       # 权重分配：人脸优先使用肤色，背景禁止抢占
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if is_face:
           if name in ["Flesh", "Tan", "White"]: dist *= 0.4 # 极大权重
       else:
           if name in ["Flesh", "Tan"]: dist *= 10.0 # 背景严禁抢肤色
           if name == "White": dist *= 2.0 # 背景尽量避开纯白
       if dist < best_dist:
           best_dist, best_key = dist, name
   inv[best_key][1] -= 1
   return inv[best_key][0], best_key
# 3. 界面布局
st.sidebar.header("🎨 艺术控制面板")
brightness = st.sidebar.slider("亮度", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("对比度 (五官锐度)", 0.5, 2.5, 1.5)
dither = st.sidebar.slider("面部颗粒混色度", 0.0, 1.0, 0.4)
zoom = st.sidebar.slider("对焦缩放", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传人像", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 预处理
   img = Image.open(uploaded_file).convert("RGB")
   img = ImageEnhance.Brightness(img).enhance(brightness)
   img = ImageEnhance.Contrast(img).enhance(contrast)
   # 强制裁剪为正方形
   w, h = img.size
   side = int(min(w, h) / zoom)
   left, top = (w - side) // 2, (h - side) // 2
   img_cropped = img.crop((left, top, left + side, top + side))
   # 缩放为 48x48 乐高网格
   small = img_cropped.resize((48, 48), Image.Resampling.LANCZOS)
   pixels = np.array(small, dtype=float) # 使用 float 防止计算溢出
   # 简单的中心人脸识别 (针对 48x48 优化)
   # 人脸通常位于图像中心 60% 区域
   face_range = range(int(48*0.2), int(48*0.8))
   current_inv = {k: [list(v[0]), v[1]] for k, v in get_inventory().items()}
   canvas = np.zeros((48, 48, 3), dtype=np.uint8)
   # 渲染像素
   for y in range(48):
       for x in range(48):
           is_face = (y in face_range and x in face_range)
           # 进行匹配
           rgb, _ = find_color(pixels[y, x], current_inv, is_face, dither)
           canvas[y, x] = rgb
   # 显示结果
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="处理预览", use_container_width=True)
   with col2:
       final_img = Image.fromarray(canvas)
       st.image(final_img.resize((600, 600), Image.Resampling.NEAREST), caption="乐高艺术转换 (无乱码)", use_container_width=True)
   # 精确统计
   with st.expander("📊 零件消耗详单"):
       raw = get_inventory()
       stats = [{"颜色": k, "已用": raw[k][1]-v[1], "剩余": v[1]} for k, v in current_inv.items()]
       st.table(stats)
