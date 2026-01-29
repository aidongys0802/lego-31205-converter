import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
# --- 1. 基础配置 ---
st.set_page_config(page_title="LEGO 31205 高精细版", layout="wide")
LEGO_31205_DATA = {
   "Black": [(0, 0, 0), 600], "Dark Stone Grey": [(99, 95, 97), 470],
   "Medium Stone Grey": [(150, 152, 152), 370], "White": [(255, 255, 255), 350],
   "Navy Blue": [(27, 42, 52), 310], "Blue": [(0, 85, 191), 280],
   "Medium Azure": [(0, 174, 216), 170], "Light Aqua": [(188, 225, 233), 140],
   "Tan": [(217, 187, 123), 140], "Flesh (Light Nougat)": [(255, 158, 146), 140],
   "Dark Orange": [(168, 84, 9), 110], "Reddish Brown": [(127, 51, 26), 100],
   "Red": [(215, 0, 0), 100], "Medium Lavender": [(156, 124, 204), 100],
   "Sand Blue": [(112, 129, 154), 100]
}
@st.cache_resource
def load_face_cascade():
   return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# --- 2. 核心算法 ---
def get_closest_color_info(target_rgb, inventory):
   """找到最接近的颜色，返回 ((r,g,b), name)"""
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_key = "Black"
   # 这里我们只做查找，不扣库存，库存最后统一扣，防止抖动计算时过度消耗
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           dist = (r - tr)**2 + (g - tg)**2 + (b - tb)**2
           if dist < best_dist:
               best_dist = dist
               best_key = name
   return inventory[best_key][0], best_key
def smart_crop(pil_img, zoom_level=1.5):
   """智能人脸裁剪：基于人脸位置进行 Zoom In"""
   face_cascade = load_face_cascade()
   # 转灰度检测
   cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
   gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   w, h = pil_img.size
   if len(faces) > 0:
       # 取最大的人脸
       faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
       fx, fy, fw, fh = faces[0]
       # 计算人脸中心
       cx, cy = fx + fw // 2, fy + fh // 2
       # 决定裁剪框大小 (基于人脸大小放大)
       # zoom_level 越小，裁剪框相对于人脸越大（也就是画面包含越多背景）
       # zoom_level 越大，画面越聚焦人脸
       crop_dim = int(max(fw, fh) * zoom_level)
       # 确保裁剪框不超出原图边界
       x1 = max(0, cx - crop_dim // 2)
       y1 = max(0, cy - crop_dim // 2)
       x2 = min(w, cx + crop_dim // 2)
       y2 = min(h, cy + crop_dim // 2)
       # 如果计算出的框不是正方形，修正它
       real_w = x2 - x1
       real_h = y2 - y1
       final_dim = min(real_w, real_h)
       return pil_img.crop((x1, y1, x1+final_dim, y1+final_dim))
   else:
       # 没检测到脸，回退到中心裁剪
       dim = min(w, h)
       left = (w - dim) // 2
       top = (h - dim) // 2
       return pil_img.crop((left, top, left + dim, top + dim))
def apply_dithering(pixel_array, size, inventory, use_dithering=True):
   """应用 Floyd-Steinberg 抖动算法"""
   h, w, _ = pixel_array.shape
   # 转换为 float 类型以处理误差扩散
   buffer = pixel_array.astype(float)
   output = np.zeros_like(pixel_array)
   stats = {}
   # 临时库存副本，用于动态检查
   temp_inv = {k: [list(v[0]), v[1]] for k, v in inventory.items()}
   for y in range(h):
       for x in range(w):
           old_pixel = buffer[y, x]
           # 1. 找到最近似颜色
           new_pixel, name = get_closest_color_info(old_pixel, temp_inv)
           # 记录使用情况
           output[y, x] = new_pixel
           stats[name] = stats.get(name, 0) + 1
           temp_inv[name][1] -= 1 # 简单扣除
           if use_dithering:
               quant_error = old_pixel - new_pixel
               # 2. 扩散误差给周围像素
               if x + 1 < w:
                   buffer[y, x + 1] += quant_error * 7 / 16
               if y + 1 < h:
                   if x - 1 >= 0:
                       buffer[y + 1, x - 1] += quant_error * 3 / 16
                   buffer[y + 1, x] += quant_error * 5 / 16
                   if x + 1 < w:
                       buffer[y + 1, x + 1] += quant_error * 1 / 16
   return output, stats
# --- 3. 界面逻辑 ---
st.title("🧩 LEGO 31205 高精细人像生成器")
with st.sidebar:
   st.header("🎛️ 精细度控制")
   grid_size = st.select_slider("画布分辨率", options=[32, 48, 64, 96], value=48)
   st.subheader("1. 构图优化")
   enable_smart_crop = st.checkbox("启用智能人脸特写 (Smart Zoom)", value=True)
   zoom_factor = st.slider("视野范围 (越小脸越大)", 1.2, 4.0, 2.5, help="数值越小，裁剪框越贴近脸部边缘")
   st.subheader("2. 细节增强")
   contrast = st.slider("对比度增强", 0.8, 2.0, 1.2)
   sharpness = st.slider("锐化程度", 0.0, 2.0, 1.3)
   st.subheader("3. 纹理质感")
   use_dithering = st.checkbox("开启颜色抖动 (Dithering)", value=True, help="混合像素以模拟更多过渡色，让皮肤更自然")
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 1. 加载与预处理
   original = Image.open(uploaded_file).convert("RGB")
   # 增强对比度和锐度
   enhancer = ImageEnhance.Contrast(original)
   img_contrast = enhancer.enhance(contrast)
   enhancer = ImageEnhance.Sharpness(img_contrast)
   img_sharp = enhancer.enhance(sharpness)
   # 2. 裁剪
   if enable_smart_crop:
       img_cropped = smart_crop(img_sharp, zoom_level=zoom_factor)
       crop_msg = "智能特写"
   else:
       w, h = img_sharp.size
       dim = min(w, h)
       left = (w - dim) // 2
       top = (h - dim) // 2
       img_cropped = img_sharp.crop((left, top, left + dim, top + dim))
       crop_msg = "居中裁剪"
   # 3. 缩放
   img_small = img_cropped.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
   pixel_data = np.array(img_small)
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption=f"处理后输入图 ({crop_msg})", use_container_width=True)
   if st.button("生成高精细乐高画"):
       # 4. 颜色量化与抖动
       final_array, usage = apply_dithering(pixel_data, grid_size, LEGO_31205_DATA, use_dithering)
       result_img = Image.fromarray(final_array.astype('uint8'))
       with col2:
           st.image(result_img.resize((600, 600), Image.Resampling.NEAREST),
                    caption="最终效果 (Nearest Neighbor 预览)", use_container_width=True)
       # 5. 统计
       st.success("生成完毕！")
       with st.expander("查看零件消耗清单"):
           sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
           st.table([{"零件颜色": k, "使用数量": v, "库存剩余": LEGO_31205_DATA[k][1]-v} for k, v in sorted_usage])
