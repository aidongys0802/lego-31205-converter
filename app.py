import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
st.set_page_config(page_title="LEGO 31205 皮肤无灰版", layout="wide")
# 1. LEGO 31205 严格库存
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
# 2. 优化后的颜色匹配算法
def find_best_match_skin_safe(target_pixel, inv, is_face):
   tr, tg, tb = target_pixel
   brightness = (tr + tg + tb) / 3.0
   best_dist = float('inf')
   best_name = None
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       if is_face:
           # --- 面部保护逻辑 ---
           # 1. 强制奖励 Tan 和 White
           if name == "White": dist *= 0.4
           elif name == "Tan": dist *= 0.5
           # 2. Flesh 仅用于表现暖色调/妆容
           elif name == "Flesh": dist *= 0.8
           # 3. 严格限制灰色：除非像素非常暗(阴影极深)，否则给灰色施加巨额惩罚
           if "Stone Grey" in name:
               if brightness > 80: # 中浅度区域严禁变灰
                   dist *= 20.0
               else: # 只有极深阴影才允许少量灰色
                   dist *= 1.5
           # 4. 黑色仅限最深线条
           if name == "Black":
               if brightness > 50: dist *= 10.0
       else:
           # --- 背景避让逻辑 ---
           if name in ["Flesh", "Tan", "White"]: dist *= 50.0 # 禁区
           if name in ["Light Aqua", "Medium Stone Grey", "Sand Blue"]: dist *= 0.5 # 推荐
       if dist < best_dist:
           best_dist = dist
           best_name = name
   if best_name:
       inv[best_name][1] -= 1
       return inv[best_name][0], best_name
   return (0, 0, 0), "Black"
# --- UI ---
st.sidebar.header("🎨 皮肤质感微调")
brightness_val = st.sidebar.slider("1. 整体亮度", 0.5, 2.0, 1.4, help="建议调高，迫使算法多用White")
contrast_val = st.sidebar.slider("2. 对比度", 0.5, 3.0, 1.8, help="调高可强化五官阴影，减少灰色过渡")
dither_strength = st.sidebar.slider("3. 抖动程度", 0.0, 1.0, 0.85)
zoom_val = st.sidebar.slider("4. 对焦", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   img = Image.open(uploaded_file).convert("RGB")
   img = ImageEnhance.Brightness(img).enhance(brightness_val)
   img = ImageEnhance.Contrast(img).enhance(contrast_val)
   w, h = img.size
   crop_dim = int(min(w, h) / zoom_val)
   cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   faces = face_cascade.detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   cx, cy = (fx + fw//2, fy + fh//2) if len(faces) > 0 else (w//2, h//2)
   left, top = max(0, min(w-crop_dim, cx-crop_dim//2)), max(0, min(h-crop_dim, cy-crop_dim//2))
   img_cropped = img.crop((left, top, left+crop_dim, top+crop_dim))
   small = img_cropped.resize((48, 48), Image.Resampling.LANCZOS)
   pixel_buffer = np.array(small, dtype=float)
   # 检测面部 Mask
   small_cv = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)
   small_faces = face_cascade.detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((48, 48), dtype=bool)
   if len(small_faces) > 0:
       for (fx, fy, fw, fh) in small_faces:
           face_mask[fy:fy+fh, fx:fx+fw] = True
   else:
       face_mask[12:36, 12:36] = True # 默认中间区域
   current_inv = get_inventory()
   canvas = np.zeros((48, 48, 3), dtype=np.uint8)
   # 全图 Floyd-Steinberg 抖动
   for y in range(48):
       for x in range(48):
           old_val = np.clip(pixel_buffer[y, x], 0, 255)
           new_rgb, _ = find_best_match_skin_safe(old_val, current_inv, face_mask[y, x])
           canvas[y, x] = new_rgb
           error = (old_val - new_rgb) * dither_strength
           if x + 1 < 48: pixel_buffer[y, x + 1] += error * 7 / 16
           if y + 1 < 48:
               if x - 1 >= 0: pixel_buffer[y + 1, x - 1] += error * 3 / 16
               pixel_buffer[y + 1, x] += error * 5 / 16
               if x + 1 < 48: pixel_buffer[y + 1, x + 1] += error * 1 / 16
   col1, col2 = st.columns(2)
   with col1: st.image(img_cropped, use_container_width=True)
   with col2: st.image(Image.fromarray(canvas).resize((600, 600), Image.Resampling.NEAREST), use_container_width=True)
   with st.expander("📊 零件消耗"):
       raw = get_inventory()
       st.table([{"颜色": k, "已用": raw[k][1]-v[1], "剩余": v[1]} for k, v in current_inv.items()])
