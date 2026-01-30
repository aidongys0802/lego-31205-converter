import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
st.set_page_config(page_title="LEGO 31205 终极质感版", layout="wide")
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
# 2. 核心算法：带区域权重的加权距离计算
def find_best_match_weighted(target_pixel, inv, is_face):
   tr, tg, tb = target_pixel
   best_dist = float('inf')
   best_name = None
   for name, (rgb, count) in inv.items():
       if count <= 0: continue
       # 1. 基础色彩距离 (加权欧氏距离，人眼对绿色更敏感)
       dr, dg, db = rgb[0] - tr, rgb[1] - tg, rgb[2] - tb
       dist = 2*dr**2 + 4*dg**2 + 3*db**2
       # 2. 区域权重策略 (核心魔法)
       if is_face:
           # --- 面部策略 ---
           # 极度鼓励 Tan 和 White 混合，模仿参考图的高光质感
           if name == "White": dist *= 0.5   # 疯狂打折
           elif name == "Tan": dist *= 0.6   # 打折
           elif name == "Flesh": dist *= 0.9 # 稍微优先，做妆容
           # 鼓励深色用于五官勾勒
           elif name in ["Black", "Dark Stone Grey", "Reddish Brown"]: dist *= 1.0
           # 惩罚冷色调，防止脸发青
           elif name in ["Light Aqua", "Blue", "Medium Azure"]: dist *= 2.0
       else:
           # --- 背景策略 ---
           # 严禁背景抢走宝贵的皮肤积木
           if name in ["Flesh", "Tan"]: dist *= 50.0
           # 背景尽量不用白色，除非万不得已
           if name == "White": dist *= 10.0
           # 强制背景倾向于冷色调浅色 (Light Aqua, Grey)
           if name in ["Light Aqua", "Medium Stone Grey", "Sand Blue"]: dist *= 0.6
       if dist < best_dist:
           best_dist = dist
           best_name = name
   if best_name:
       inv[best_name][1] -= 1
       return inv[best_name][0], best_name
   return (0, 0, 0), "Black"
# --- UI ---
st.sidebar.header("🎨 参考图复刻调节")
st.sidebar.markdown("**核心调节指南：**\n* 想要图例那种白点多的效果，请调高亮度。\n* 想要五官清晰，请调高对比度。")
brightness_val = st.sidebar.slider("1. 亮度 (控制高光White占比)", 0.5, 2.5, 1.3)
contrast_val = st.sidebar.slider("2. 对比度 (控制Tan/White分离)", 0.5, 3.0, 1.6)
dither_strength = st.sidebar.slider("3. 抖动强度 (颗粒感)", 0.0, 1.0, 0.9, help="越接近1.0，颗粒感越强，越像参考图")
zoom_val = st.sidebar.slider("4. 人脸缩放", 1.0, 3.0, 1.8)
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # A. 图像增强
   img = Image.open(uploaded_file).convert("RGB")
   img = ImageEnhance.Brightness(img).enhance(brightness_val)
   img = ImageEnhance.Contrast(img).enhance(contrast_val)
   # B. 智能裁剪 (保持正方形)
   w, h = img.size
   crop_dim = int(min(w, h) / zoom_val)
   # 尝试检测人脸以中心对齐
   cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   faces = face_cascade.detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
   if len(faces) > 0:
       fx, fy, fw, fh = max(faces, key=lambda x: x[2]*x[3]) # 取最大人脸
       cx, cy = fx + fw//2, fy + fh//2
   else:
       cx, cy = w//2, h//2
   left = max(0, min(w - crop_dim, cx - crop_dim // 2))
   top = max(0, min(h - crop_dim, cy - crop_dim // 2))
   img_cropped = img.crop((left, top, left + crop_dim, top + crop_dim))
   # C. 缩放至 48x48 并转为 Float 进行计算
   small = img_cropped.resize((48, 48), Image.Resampling.LANCZOS)
   pixel_buffer = np.array(small, dtype=float)
   # D. 生成 Mask (再次在小图上确认人脸区域)
   small_cv = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR)
   small_faces = face_cascade.detectMultiScale(cv2.cvtColor(small_cv, cv2.COLOR_BGR2GRAY), 1.05, 1)
   face_mask = np.zeros((48, 48), dtype=bool)
   if len(small_faces) > 0:
       for (fx, fy, fw, fh) in small_faces:
           # 稍微扩大一点 mask 范围，保证脸颊边缘也被覆盖
           pad = 1
           face_mask[max(0,fy-pad):min(48,fy+fh+pad), max(0,fx-pad):min(48,fx+fw+pad)] = True
   else:
       # 兜底：如果没有检测到，假设中间 50% 是脸
       face_mask[12:36, 12:36] = True
   # E. Floyd-Steinberg 误差扩散循环
   current_inv = get_inventory()
   canvas = np.zeros((48, 48, 3), dtype=np.uint8)
   for y in range(48):
       for x in range(48):
           # 1. 读取当前像素 (含之前传递过来的误差)
           old_val = np.clip(pixel_buffer[y, x], 0, 255)
           # 2. 寻找最佳积木 (应用区域权重)
           is_face_pixel = face_mask[y, x]
           new_rgb, name = find_best_match_weighted(old_val, current_inv, is_face_pixel)
           # 3. 填入画布
           canvas[y, x] = new_rgb
           # 4. 计算误差
           error = (old_val - new_rgb) * dither_strength
           # 5. 扩散误差 (Floyd-Steinberg 矩阵)
           #       X   7
           #   3   5   1
           if x + 1 < 48:
               pixel_buffer[y, x + 1] += error * 7 / 16
           if y + 1 < 48:
               if x - 1 >= 0:
                   pixel_buffer[y + 1, x - 1] += error * 3 / 16
               pixel_buffer[y + 1, x] += error * 5 / 16
               if x + 1 < 48:
                   pixel_buffer[y + 1, x + 1] += error * 1 / 16
   # F. 展示结果
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="裁切预览", use_container_width=True)
   with col2:
       res_img = Image.fromarray(canvas)
       st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), caption="最终效果", use_container_width=True)
   # G. 消耗统计
   with st.expander("📊 31205 库存实时监控"):
       raw_inv = get_inventory()
       stats = []
       # 分类展示
       face_colors = ["White", "Tan", "Flesh", "Dark Orange"]
       bg_colors = ["Light Aqua", "Medium Stone Grey", "Sand Blue", "Blue"]
       st.write("**核心肤色消耗:**")
       cols = st.columns(len(face_colors))
       for idx, k in enumerate(face_colors):
           used = raw_inv[k][1] - current_inv[k][1]
           cols[idx].metric(k, f"{used}/{raw_inv[k][1]}", delta=current_inv[k][1])
       st.write("**背景替代色消耗:**")
       cols2 = st.columns(len(bg_colors))
       for idx, k in enumerate(bg_colors):
           used = raw_inv[k][1] - current_inv[k][1]
           cols2[idx].metric(k, f"{used}/{raw_inv[k][1]}", delta=current_inv[k][1])
