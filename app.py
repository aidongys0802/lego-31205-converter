import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math
# --- 1. 配置与乐高 31205 数据 ---
st.set_page_config(page_title="LEGO 31205 人像转换器 (OpenCV 稳定版)", layout="wide")
# 乐高 31205 (蝙蝠侠) 零件列表：颜色名称 -> [(R, G, B), 数量]
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
# 使用 OpenCV 自带的人脸检测模型
@st.cache_resource
def load_face_cascade():
   # 获取 OpenCV 自带的分类器路径
   cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
   return cv2.CascadeClassifier(cascade_path)
# --- 2. 核心算法逻辑 ---
def get_closest_available(target_rgb, inventory):
   r, g, b = target_rgb
   candidates = []
   for name, data in inventory.items():
       rgb, count = data
       if count > 0:
           # 计算欧式距离，找到最接近的颜色
           dist = math.sqrt((r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2)
           candidates.append((dist, name))
   if not candidates:
       return (0, 0, 0), "Black" # 如果所有颜色的库存都用光了，返回黑色
   candidates.sort()
   best_name = candidates[0][1]
   inventory[best_name][1] -= 1 # 消耗一个零件
   return inventory[best_name][0], best_name
def process_image(pil_img, size, p_weights):
   face_cascade = load_face_cascade()
   # 转换为 RGB 和 Gray
   img_rgb = np.array(pil_img.convert("RGB"))
   # 计算裁剪坐标，将图片中心裁剪为正方形
   h, w, _ = img_rgb.shape
   crop_size = min(h, w)
   y0, x0 = (h - crop_size)//2, (w - crop_size)//2
   # 裁剪并缩放
   cropped_rgb = img_rgb[y0:y0+crop_size, x0:x0+crop_size]
   # 在裁剪后的图片上进行人脸检测
   cropped_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
   faces = face_cascade.detectMultiScale(cropped_gray, 1.1, 4)
   # 将图片缩放到画布尺寸（例如 48x48）
   # OpenCV resize 默认输出 BGR 格式
   img_s_bgr = cv2.resize(cropped_rgb, (size, size), interpolation=cv2.INTER_AREA)
   # --- 关键修复：将 BGR 转换为 RGB ---
   img_s_rgb = cv2.cvtColor(img_s_bgr, cv2.COLOR_BGR2RGB)
   # 用于计算亮度的 HSV 图像
   img_hsv = cv2.cvtColor(img_s_rgb, cv2.COLOR_RGB2HSV)
   pixel_tasks = []
   for y in range(size):
       for x in range(size):
           # 将当前像素坐标映射回原图比例，判断是否在脸部框内
           rel_x, rel_y = (x / size) * crop_size, (y / size) * crop_size
           is_face = False
           for (fx, fy, fw, fh) in faces:
               if fx <= rel_x <= fx + fw and fy <= rel_y <= fy + fh:
                   is_face = True
                   break
           v_val = img_hsv[y, x, 2] # 获取亮度值 V
           if is_face:
               score = p_weights['face']
           elif v_val > 200:
               score = p_weights['bg_high']
           elif v_val < 50:
               score = p_weights['bg_dark']
           else:
               score = p_weights['bg_normal']
           # 这里的 img_s_rgb[y, x] 现在是正确的 RGB 颜色
           pixel_tasks.append({'pos':(x,y), 'rgb':img_s_rgb[y,x], 'score':score})
   # 根据优先级排序，优先分配重要区域的颜色
   pixel_tasks.sort(key=lambda t: t['score'], reverse=True)
   # 复制一份库存数据用于计算
   curr_inv = {k: [v[0], v[1]] for k, v in LEGO_31205_DATA.items()}
   res_pixels = {}
   usage = {}
   for task in pixel_tasks:
       rgb, name = get_closest_available(task['rgb'], curr_inv)
       res_pixels[task['pos']] = rgb
       usage[name] = usage.get(name, 0) + 1
   # 生成最终的乐高预览图
   out_img = Image.new("RGB", (size, size))
   pix = out_img.load()
   for pos, rgb in res_pixels.items():
       pix[pos[0], pos[1]] = tuple(map(int, rgb))
   return out_img, usage
# --- 3. 界面布局 (保持一致) ---
st.title("🧩 LEGO 31205 艺术画转换器 (OpenCV 稳定版)")
with st.sidebar:
   st.header("⚙️ 参数设置")
   grid_size = st.slider("画布尺寸 (颗粒数)", 16, 128, 48)
   w_face = st.number_input("人物面部优先级", value=2000)
   w_high = st.number_input("背景高光优先级", value=500)
   w_normal = st.number_input("背景普通优先级", value=200)
   w_dark = st.number_input("背景阴影优先级", value=100)
uploaded_file = st.file_uploader("选择照片...", type=["jpg", "jpeg", "png"])
if uploaded_file:
   image = Image.open(uploaded_file)
   col1, col2 = st.columns(2)
   with col1:
       st.image(image, caption="原始照片", use_container_width=True)
   if st.button("生成乐高画"):
       p_weights = {'face': w_face, 'bg_high': w_high, 'bg_normal': w_normal, 'bg_dark': w_dark}
       result_img, usage_stats = process_image(image, grid_size, p_weights)
       with col2:
           # 使用 Nearest Neighbor 插值放大，保持像素感
           st.image(result_img.resize((600, 600), resample=0), caption="预览", use_container_width=True)
       st.subheader("📊 零件消耗")
       cols = st.columns(3)
       for i, (name, count) in enumerate(usage_stats.items()):
           original_stock = LEGO_31205_DATA[name][1]
           cols[i % 3].metric(name, f"{count} 颗", f"剩余 {original_stock - count}")
