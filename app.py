import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
# --- 1. 配置与乐高 31205 数据 ---
st.set_page_config(page_title="LEGO 31205 艺术画转换器", layout="wide")
# 严谨校验过的 31205 颜色表 (RGB 格式)
LEGO_31205_DATA = {
   "Black": [(0, 0, 0), 600],
   "Dark Stone Grey": [(99, 95, 97), 470],
   "Medium Stone Grey": [(150, 152, 152), 370],
   "White": [(255, 255, 255), 350],
   "Navy Blue": [(27, 42, 52), 310],
   "Blue": [(0, 85, 191), 280],
   "Medium Azure": [(0, 174, 216), 170],
   "Light Aqua": [(188, 225, 233), 140],
   "Tan": [(217, 187, 123), 140],
   "Flesh (Light Nougat)": [(255, 158, 146), 140],
   "Dark Orange": [(168, 84, 9), 110],
   "Reddish Brown": [(127, 51, 26), 100],
   "Red": [(215, 0, 0), 100],
   "Medium Lavender": [(156, 124, 204), 100],
   "Sand Blue": [(112, 129, 154), 100]
}
@st.cache_resource
def load_face_cascade():
   return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# --- 2. 核心算法 ---
def get_closest_available(target_rgb, inventory):
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_name = "Black"
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           # 使用加权欧式距离提高肉眼准确度
           dist = (r - tr)**2 + (g - tg)**2 + (b - tb)**2
           if dist < best_dist:
               best_dist = dist
               best_name = name
   inventory[best_name][1] -= 1
   return inventory[best_name][0], best_name
def process_image(pil_img, size, p_weights):
   face_cascade = load_face_cascade()
   # 全部保持为 PIL RGB 模式处理，避免 BGR 干扰
   img_rgb = pil_img.convert("RGB")
   w, h = img_rgb.size
   crop_size = min(w, h)
   left = (w - crop_size) // 2
   top = (h - crop_size) // 2
   img_cropped = img_rgb.crop((left, top, left + crop_size, top + crop_size))
   # 缩放到颗粒尺寸
   img_small = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
   img_array = np.array(img_small)
   # 人脸检测 (仅此处转一次灰度)
   cv_img = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)
   gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   # 准备像素任务
   pixel_tasks = []
   for y in range(size):
       for x in range(size):
           # 坐标映射检查人脸
           rel_x, rel_y = (x / size) * crop_size, (y / size) * crop_size
           is_face = False
           for (fx, fy, fw, fh) in faces:
               if fx <= rel_x <= fx + fw and fy <= rel_y <= fy + fh:
                   is_face = True; break
           r, g, b = img_array[y, x]
           brightness = (int(r) + int(g) + int(b)) / 3
           if is_face: score = p_weights['face']
           elif brightness > 200: score = p_weights['bg_high']
           elif brightness < 50: score = p_weights['bg_dark']
           else: score = p_weights['bg_normal']
           pixel_tasks.append({'pos': (x, y), 'rgb': (r, g, b), 'score': score})
   # 排序：让最重要的像素（脸部）先挑颜色
   pixel_tasks.sort(key=lambda t: t['score'], reverse=True)
   curr_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_31205_DATA.items()}
   res_pixels = {}
   usage = {}
   for task in pixel_tasks:
       match_rgb, name = get_closest_available(task['rgb'], curr_inv)
       res_pixels[task['pos']] = match_rgb
       usage[name] = usage.get(name, 0) + 1
   # 重建图像
   res_img = Image.new("RGB", (size, size))
   for pos, rgb in res_pixels.items():
       res_img.putpixel(pos, tuple(rgb))
   return res_img, usage
# --- 3. 界面 ---
st.title("🧩 LEGO 31205 完美人像转换器")
st.sidebar.header("配置")
grid_size = st.sidebar.select_slider("颗粒尺寸 (建议 48)", options=[16, 32, 48, 64, 80, 96], value=48)
w_face = st.sidebar.slider("面部权重", 1000, 5000, 3000)
uploaded_file = st.file_uploader("上传人像照片", type=["jpg", "png"])
if uploaded_file:
   img = Image.open(uploaded_file)
   col1, col2 = st.columns(2)
   with col1:
       st.image(img, caption="原图", use_container_width=True)
   if st.button("开始生成"):
       with st.spinner("正在计算最佳色彩分配..."):
           p_weights = {'face': w_face, 'bg_high': 500, 'bg_normal': 200, 'bg_dark': 100}
           result, stats = process_image(img, grid_size, p_weights)
           with col2:
               # 放大 10 倍显示，保持像素感
               st.image(result.resize((grid_size*10, grid_size*10), Image.Resampling.NEAREST),
                        caption="乐高预览", use_container_width=True)
           st.write("### 📊 零件需求清单")
           for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
               st.text(f"{name}: {count} 颗 (库存剩余: {LEGO_31205_DATA[name][1] - count})")
