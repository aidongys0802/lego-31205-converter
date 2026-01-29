import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
st.set_page_config(page_title="LEGO 31205 艺术大师版", layout="wide")
# 1. 严格 31205 零件库及库存
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
   # 加载 OpenCV 人脸检测器
   return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_best_color(target_rgb, current_inv, is_face_roi=False, dither_rate=0.0):
   """
   核心配色逻辑：
   - target_rgb: 目标像素值
   - is_face_roi: 是否为人脸区域
   - dither_rate: 稀疏抖动频率（用于表现皮肤质感）
   """
   tr, tg, tb = [int(np.clip(c, 0, 255)) for c in target_rgb]
   # 定义人脸核心色板
   skin_base = ["Flesh", "White", "Tan", "Dark Orange", "Reddish Brown", "Black"]
   # 定义妆造/艺术点缀色板
   makeup_colors = ["Red", "Medium Lavender", "Medium Azure", "Blue", "Light Aqua", "Navy Blue"]
   # 皮肤随机高光处理 (Flesh 混 White)
   if is_face_roi and dither_rate > 0 and random.random() < dither_rate:
       tr, tg, tb = [min(255, c + 35) for c in [tr, tg, tb]]
   # 通过饱和度识别妆造（如红唇、彩妆）
   max_c, min_c = max(tr, tg, tb), min(tr, tg, tb)
   sat = (max_c - min_c) / (max_c + 0.1)
   best_dist = float('inf')
   best_name = "Black"
   for name, (rgb, count) in current_inv.items():
       if count <= 0: continue
       if is_face_roi:
           # 逻辑：低饱和度区域强制使用肤色，高饱和度区域允许使用彩色积木表现妆造
           if sat < 0.25 and name not in skin_base: continue
           if sat >= 0.25 and name not in (skin_base + makeup_colors): continue
       # 加权感知色彩距离 (人眼对绿色更敏感)
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if dist < best_dist:
           best_dist, best_name = dist, name
   current_inv[best_name][1] -= 1
   return current_inv[best_name][0], best_name
# --- 侧边栏所有控制滑块 ---
st.sidebar.header("🎛️ 图像增强控制")
brightness = st.sidebar.slider("1. 整体亮度 (提亮皮肤)", 0.5, 2.0, 1.1)
contrast = st.sidebar.slider("2. 五官锐度 (对比度)", 0.5, 2.5, 1.4)
skin_dither = st.sidebar.slider("3. 皮肤质感点 (稀疏抖动)", 0.0, 0.4, 0.05)
st.sidebar.header("📐 构图设置")
zoom = st.sidebar.slider("对焦范围 (Zoom)", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片开始转换", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # A. 图像加载与基础增强
   img_pil = Image.open(uploaded_file).convert("RGB")
   img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
   img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
   # B. 人脸定位与智能裁剪
   cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
   faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   w, h = img_pil.size
   grid_res = 48 # 固定为 31205 标准尺寸
   if len(faces) > 0:
       # 选择画面中最大的人脸进行对焦
       fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
       cx, cy = fx + fw//2, fy + fh//2
       crop_dim = int(min(w, h) / zoom)
       left, top = max(0, cx - crop_dim//2), max(0, cy - crop_dim//2)
       img_cropped = img_pil.crop((left, top, min(w, left+crop_dim), min(h, top+crop_dim)))
   else:
       # 无人脸时居中裁剪
       dim = min(w, h)
       img_cropped = img_pil.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))
   # C. 核心像素映射逻辑
   # 缩放至 48x48 乐高网格
   img_small = img_cropped.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
   pixel_array = np.array(img_small)
   # 在小图上再次确定人脸 ROI，防止背景干扰
   small_cv = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
   small_faces = load_cascade().detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((grid_res, grid_res), dtype=bool)
   if len(small_faces) > 0:
       for (sx, sy, sw, sh) in small_faces:
           face_mask[sy:sy+sh, sx:sx+sw] = True
   # 初始化运行库存副本
   run_inv = {k: [list(v[0]), v[1]] for k, v in LEGO_INVENTORY.items()}
   canvas = np.zeros((grid_res, grid_res, 3), dtype=np.uint8)
   # 渲染每一个网格点
   for y in range(grid_res):
       for x in range(grid_res):
           is_face_pixel = face_mask[y, x]
           rgb, _ = get_best_color(
               pixel_array[y, x],
               run_inv,
               is_face_roi=is_face_pixel,
               dither_rate=skin_dither
           )
           canvas[y, x] = rgb
   # D. 结果展示
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="图像预处理预览", use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       # 展示放大后的像素效果
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST),
                caption="乐高 48x48 艺术转换结果", use_container_width=True)
   # E. 库存实时监控表
   with st.expander("📊 查看详细零件消耗 (严格基于 31205 库存)"):
       stats = []
       for name, original in LEGO_INVENTORY.items():
           used = original[1] - run_inv[name][1]
           stats.append({"颜色": name, "已用数量": used, "库内剩余": run_inv[name][1]})
       st.table(stats)
