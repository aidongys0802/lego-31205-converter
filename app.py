import streamlit as st
import numpy as np
import cv2
from PIL import Image
# --- 1. 核心数据 ---
st.set_page_config(page_title="LEGO 31205 艺术画转换器", layout="wide")
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
# --- 2. 算法逻辑 ---
def find_best_color(target_rgb, inventory):
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_name = None
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           # 使用简单的颜色距离
           dist = (r - tr)**2 + (g - tg)**2 + (b - tb)**2
           if dist < best_dist:
               best_dist = dist
               best_name = name
   if best_name:
       inventory[best_name][1] -= 1
       return inventory[best_name][0], best_name
   return (0, 0, 0), "Black"
def process_image(pil_img, size):
   face_cascade = load_face_cascade()
   # 统一转为 RGB 模式
   img_rgb = pil_img.convert("RGB")
   w, h = img_rgb.size
   crop_size = min(w, h)
   left, top = (w - crop_size) // 2, (h - crop_size) // 2
   img_cropped = img_rgb.crop((left, top, left + crop_size, top + crop_size))
   # 缩小到目标格子数
   img_small = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
   pixel_data = np.array(img_small) # [y, x, c]
   # 人脸检测
   cv_img = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)
   gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   # 1. 标记像素类型
   pixel_info = []
   for y in range(size):
       row = []
       for x in range(size):
           rel_x, rel_y = (x / size) * crop_size, (y / size) * crop_size
           is_face = False
           for (fx, fy, fw, fh) in faces:
               if fx <= rel_x <= fx + fw and fy <= rel_y <= fy + fh:
                   is_face = True; break
           row.append(is_face)
       pixel_info.append(row)
   # 2. 分配积木 (按优先级：先脸部，后其他)
   curr_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_31205_DATA.items()}
   # 创建一个空的画布数组
   canvas = np.zeros((size, size, 3), dtype=np.uint8)
   usage = {}
   # 第一遍：处理脸部
   for y in range(size):
       for x in range(size):
           if pixel_info[y][x]:
               best_rgb, name = find_best_color(pixel_data[y, x], curr_inv)
               canvas[y, x] = best_rgb
               usage[name] = usage.get(name, 0) + 1
   # 第二遍：处理非脸部
   for y in range(size):
       for x in range(size):
           if not pixel_info[y][x]:
               best_rgb, name = find_best_color(pixel_data[y, x], curr_inv)
               canvas[y, x] = best_rgb
               usage[name] = usage.get(name, 0) + 1
   return Image.fromarray(canvas), usage
# --- 3. UI 界面 ---
st.title("🧩 LEGO 31205 艺术画生成器")
with st.sidebar:
   grid_size = st.slider("格子数量", 16, 64, 48)
   st.info("48x48 是 31205 套装的标准尺寸")
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   img = Image.open(uploaded_file)
   col1, col2 = st.columns(2)
   with col1:
       st.image(img, caption="原图", use_container_width=True)
   if st.button("生成"):
       result, stats = process_image(img, grid_size)
       with col2:
           # 放大预览，Nearest 保持像素点颗粒感
           st.image(result.resize((600, 600), Image.Resampling.NEAREST),
                    caption="生成的乐高画", use_container_width=True)
       st.write("### 📦 零件清单")
       st.table([{"颜色": k, "数量": v} for k, v in stats.items()])
