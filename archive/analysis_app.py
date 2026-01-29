import streamlit as st
import numpy as np
import cv2
from PIL import Image
# --- 1. 基础数据 ---
st.set_page_config(page_title="LEGO 诊断工具", layout="wide")
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
def find_best_color(target_rgb, inventory):
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_name = "Black"
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           dist = (int(r) - int(tr))**2 + (int(g) - int(tg))**2 + (int(b) - int(tb))**2
           if dist < best_dist:
               best_dist = dist
               best_name = name
   inventory[best_name][1] -= 1
   return inventory[best_name][0], best_name
# --- 界面布局 ---
st.title("🔍 LEGO 31205 转换工序诊断器")
st.markdown("通过观察以下四个阶段，我们可以精准定位问题出在哪里。")
uploaded_file = st.file_uploader("上传测试照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 阶段 1: 原始输入
   img_raw = Image.open(uploaded_file)
   col1, col2, col3, col4 = st.columns(4)
   with col1:
       st.image(img_raw, caption="1. 原始上传", use_container_width=True)
   # 阶段 2: 预处理与裁剪
   # 强制转 RGB 并进行正方形裁剪
   img_rgb = img_raw.convert("RGB")
   w, h = img_rgb.size
   crop_size = min(w, h)
   left, top = (w - crop_size) // 2, (h - crop_size) // 2
   img_cropped = img_rgb.crop((left, top, left + crop_size, top + crop_size))
   with col2:
       st.image(img_cropped, caption="2. RGB 正方形裁剪", use_container_width=True)
   # 阶段 3: 重采样 (Resize)
   grid_size = st.sidebar.slider("格子数", 16, 64, 48)
   # 使用 NEAREST 观察最原始的像素颗粒
   img_small = img_cropped.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
   # 这里的数据是后续算法的唯一输入
   pixel_data = np.array(img_small)
   with col3:
       # 放大显示，确保像素对齐
       st.image(img_small.resize((600, 600), Image.Resampling.NEAREST),
                caption="3. 缩放后的像素输入", use_container_width=True)
   # 阶段 4: 算法输出
   if st.button("运行颜色匹配诊断"):
       # 深拷贝库存
       curr_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_31205_DATA.items()}
       # 创建画布，明确 [高度, 宽度, 通道]
       canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
       # 逐像素处理，绝不排序，绝不改变坐标索引
       for y in range(grid_size):
           for x in range(grid_size):
               target_rgb = pixel_data[y, x]
               match_rgb, _ = find_best_color(target_rgb, curr_inv)
               canvas[y, x] = match_rgb
       result_img = Image.fromarray(canvas)
       with col4:
           st.image(result_img.resize((600, 600), Image.Resampling.NEAREST),
                    caption="4. 最终乐高匹配结果", use_container_width=True)
       # 检查是否出现了坐标偏移
       if pixel_data.shape[:2] != canvas.shape[:2]:
           st.error(f"维度不匹配! 输入: {pixel_data.shape}, 输出: {canvas.shape}")
