import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
st.set_page_config(page_title="LEGO 31205 艺术优化版", layout="wide")
# 1. LEGO 31205 官方零件库
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
# 2. 核心算法：高光优先 + 皮肤去灰
def find_best_match(target_pixel, inv, is_face):
   tr, tg, tb = target_pixel
   brightness = (tr + tg + tb) / 3.0
   best_dist = float('inf')
   best_name = None
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if is_face:
           # --- 脸部：高光 Tan+White 模式 ---
           if name == "White": dist *= 0.45  # 奖励白色做高光
           elif name == "Tan": dist *= 0.55  # 奖励沙色做主色
           elif name == "Flesh": dist *= 0.85 # 适度奖励肉色做妆容
           # 严禁面部在中亮区域出现灰色
           if "Stone Grey" in name:
               dist *= 15.0 if brightness > 70 else 1.2
           # 黑色仅用于极深阴影和轮廓
           if name == "Black" and brightness > 40: dist *= 8.0
       else:
           # --- 背景：冷色避让模式 ---
           if name in ["Flesh", "Tan", "White"]: dist *= 40.0 # 锁死肤色资源
           if name in ["Light Aqua", "Medium Stone Grey", "Sand Blue"]: dist *= 0.6 # 奖励替代色
       if dist < best_dist:
           best_dist, best_name = dist, name
   if best_name:
       inv[best_name][1] -= 1
       return inv[best_name][0], best_name
   return (0, 0, 0), "Black"
# --- Streamlit UI ---
st.sidebar.header("🎨 艺术颗粒调节")
bright_val = st.sidebar.slider("1. 整体亮度", 0.5, 2.0, 1.35)
cont_val = st.sidebar.slider("2. 对比度 (五官深度)", 0.5, 3.0, 1.7)
dither_val = st.sidebar.slider("3. 抖动强度 (颗粒感)", 0.0, 1.0, 0.9)
zoom_val = st.sidebar.slider("4. 人脸缩放", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传人像", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # A. 预处理
   img = Image.open(uploaded_file).convert("RGB")
   img = ImageEnhance.Brightness(img).enhance(bright_val)
   img = ImageEnhance.Contrast(img).enhance(cont_val)
   # B. 裁剪逻辑修复 (修复 fx 未定义错误)
   w, h = img.size
   cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   faces = face_cascade.detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   if len(faces) > 0:
       # 正确解包坐标
       fx, fy, fw, fh = max(faces, key=lambda x: x[2]*x[3])
       cx, cy = fx + fw//2, fy + fh//2
   else:
       cx, cy = w//2, h//2
   crop_dim = int(min(w, h) / zoom_val)
   left = max(0, min(w - crop_dim, cx - crop_dim // 2))
   top = max(0, min(h - crop_dim, cy - crop_dim // 2))
   img_cropped = img.crop((left, top, left + crop_dim, top + crop_dim))
   # C. 准备渲染
   small = img_cropped.resize((48, 48), Image.Resampling.LANCZOS)
   pixel_buf = np.array(small, dtype=float)
   # 重新在小图上做人脸 Mask 以保证精度
   small_cv = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)
   small_faces = face_cascade.detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((48, 48), dtype=bool)
   if len(small_faces) > 0:
       for (sf_x, sf_y, sf_w, sf_h) in small_faces:
           face_mask[sf_y:sf_y+sf_h, sf_x:sf_x+sf_w] = True
   else:
       face_mask[12:36, 12:36] = True # 默认中间
   # D. 全图 Floyd-Steinberg 抖动
   curr_inv = get_inventory()
   canvas = np.zeros((48, 48, 3), dtype=np.uint8)
   for y in range(48):
       for x in range(48):
           old_rgb = np.clip(pixel_buf[y, x], 0, 255)
           new_rgb, _ = find_best_match(old_rgb, curr_inv, face_mask[y, x])
           canvas[y, x] = new_rgb
           error = (old_rgb - new_rgb) * dither_val
           if x + 1 < 48: pixel_buf[y, x + 1] += error * 7/16
           if y + 1 < 48:
               if x - 1 >= 0: pixel_buf[y + 1, x - 1] += error * 3/16
               pixel_buf[y + 1, x] += error * 5/16
               if x + 1 < 48: pixel_buf[y + 1, x + 1] += error * 1/16
   # E. 结果展示
   col1, col2 = st.columns(2)
   with col1: st.image(img_cropped, use_container_width=True)
   with col2: st.image(Image.fromarray(canvas).resize((600, 600), Image.Resampling.NEAREST), use_container_width=True)
   with st.expander("📊 零件消耗详单"):
       raw = get_inventory()
       st.table([{"颜色": k, "已用": raw[k][1]-v[1], "剩余": v[1]} for k, v in curr_inv.items()])
