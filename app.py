import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
# --- 1. 基础配置 ---
st.set_page_config(page_title="LEGO 31205 最终完美版", layout="wide")
# 原始库存数据
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
# --- 2. 智能计算核心 ---
def detect_face_rect(pil_img):
   """检测人脸，返回 (x, y, w, h)"""
   face_cascade = load_face_cascade()
   cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
   gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   if len(faces) > 0:
       # 找最大的脸
       return sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
   return None
def generate_priority_mask(pil_img_cropped, grid_size):
   """生成优先级掩码：True=人脸VIP区域, False=背景"""
   face_rect = detect_face_rect(pil_img_cropped)
   mask = np.zeros((grid_size, grid_size), dtype=bool)
   if face_rect is not None:
       fx, fy, fw, fh = face_rect
       # 将原图坐标映射到 grid_size 坐标
       scale_x = grid_size / pil_img_cropped.size[0]
       scale_y = grid_size / pil_img_cropped.size[1]
       gx = int(fx * scale_x)
       gy = int(fy * scale_y)
       gw = int(fw * scale_x)
       gh = int(fh * scale_y)
       # 稍微向内收缩，确保VIP区域全是干货
       pad = 1
       mask[gy+pad : gy+gh-pad, gx+pad : gx+gw-pad] = True
   return mask
def find_best_available_color(target_rgb, inventory):
   """在有库存的颜色中找最接近的"""
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_key = None
   best_rgb = (0, 0, 0)
   # 遍历所有颜色，必须 check count > 0
   available_found = False
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           available_found = True
           # 加权欧式距离 (人眼对绿色更敏感，修正色彩偏差)
           dist = 2*(r - tr)**2 + 4*(g - tg)**2 + 3*(b - tb)**2
           if dist < best_dist:
               best_dist = dist
               best_key = name
               best_rgb = (r, g, b)
   if not available_found:
       return (0, 0, 0), "StockOut" # 理论上不应该发生，除非几千个零件全用光
   return best_rgb, best_key
def process_priority_allocation(pixel_array, priority_mask, inventory, use_dithering_bg):
   """
   两阶段分配算法：
   1. 优先满足 Priority Mask (人脸)
   2. 剩余库存满足 背景 (可选抖动)
   """
   h, w, _ = pixel_array.shape
   canvas = np.zeros_like(pixel_array)
   # 记录哪些像素已经填过了
   filled_map = np.zeros((h, w), dtype=bool)
   # 复制一份库存用于计算
   temp_inv = {k: [list(v[0]), v[1]] for k, v in inventory.items()}
   usage_stats = {}
   # --- 第一阶段：VIP 人脸通道 (绝不抖动，优先选色) ---
   for y in range(h):
       for x in range(w):
           if priority_mask[y, x]:
               target = pixel_array[y, x]
               rgb, name = find_best_available_color(target, temp_inv)
               if name != "StockOut":
                   canvas[y, x] = rgb
                   temp_inv[name][1] -= 1
                   usage_stats[name] = usage_stats.get(name, 0) + 1
                   filled_map[y, x] = True
   # --- 第二阶段：背景通道 (使用剩余库存，可选抖动) ---
   # 为了支持抖动，我们需要一个 float 类型的 buffer
   buffer = pixel_array.astype(float)
   for y in range(h):
       for x in range(w):
           # 只有没填过的才处理
           if not filled_map[y, x]:
               old_pixel = buffer[y, x]
               rgb, name = find_best_available_color(old_pixel, temp_inv)
               if name != "StockOut":
                   canvas[y, x] = rgb
                   temp_inv[name][1] -= 1
                   usage_stats[name] = usage_stats.get(name, 0) + 1
                   # 只有背景开启抖动时，且当前像素不是边缘，才扩散误差
                   if use_dithering_bg:
                       quant_error = old_pixel - rgb
                       # 误差扩散 (Floyd-Steinberg)
                       # 注意：不要把误差扩散进“人脸区域”，否则人脸边缘会脏
                       if x + 1 < w and not priority_mask[y, x+1]:
                           buffer[y, x + 1] += quant_error * 7 / 16
                       if y + 1 < h:
                           if x - 1 >= 0 and not priority_mask[y+1, x-1]:
                               buffer[y + 1, x - 1] += quant_error * 3 / 16
                           if not priority_mask[y+1, x]:
                               buffer[y + 1, x] += quant_error * 5 / 16
                           if x + 1 < w and not priority_mask[y+1, x+1]:
                               buffer[y + 1, x + 1] += quant_error * 1 / 16
   return canvas, usage_stats
# --- 3. 界面逻辑 ---
st.title("🧩 LEGO 31205 智能优先版")
st.markdown("🚀 **核心升级**：库存不足时，优先保证人脸使用最准确的积木颜色。")
# 使用 Form 解决滑块卡顿问题
with st.sidebar.form("settings_form"):
   st.header("🎛️ 参数设置")
   grid_size = st.select_slider("画布分辨率", options=[32, 48, 64], value=48)
   st.subheader("1. 图像处理")
   # 修正：Zoom Level 说明更清晰
   zoom_factor = st.slider("人脸放大倍数", 1.0, 3.0, 2.0, help="1.0=原图比例，3.0=超大特写")
   contrast = st.slider("对比度增强", 0.8, 1.8, 1.3)
   brightness = st.slider("亮度提升", 0.8, 1.5, 1.1)
   st.subheader("2. 风格化")
   use_dithering_bg = st.checkbox("背景使用纹理 (抖动)", value=True)
   # 提交按钮
   submit_btn = st.form_submit_button("🔨 生成/更新预览")
uploaded_file = st.file_uploader("上传照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 1. 预处理
   img_raw = Image.open(uploaded_file).convert("RGB")
   enhancer_bri = ImageEnhance.Brightness(img_raw)
   img_bri = enhancer_bri.enhance(brightness)
   enhancer_con = ImageEnhance.Contrast(img_bri)
   img_processed = enhancer_con.enhance(contrast)
   # 2. 智能裁剪
   face_rect_raw = detect_face_rect(img_processed)
   w, h = img_processed.size
   if face_rect_raw is not None:
       fx, fy, fw, fh = face_rect_raw
       cx, cy = fx + fw // 2, fy + fh // 2
       # 根据放大倍数计算裁剪框
       crop_dim = int(max(fw, fh) * (4.0 - zoom_factor)) # 修正逻辑：zoom越大，除数越小不太直观，改为反向逻辑适配
       # 重新写一个更直观的逻辑：
       # zoom=1.0 -> 裁剪框很大(包含背景)
       # zoom=3.0 -> 裁剪框很小(只看脸)
       base_size = max(fw, fh)
       # 限制最大裁剪框不超过原图短边
       max_crop = min(w, h)
       # 限制最小裁剪框不小于人脸
       min_crop = base_size
       # 线性插值计算实际裁剪大小
       # Slider 1.0 -> max_crop
       # Slider 3.0 -> min_crop
       t = (zoom_factor - 1.0) / 2.0 # 0.0 to 1.0
       current_crop_size = int(max_crop - t * (max_crop - min_crop))
       half_size = current_crop_size // 2
       x1 = max(0, cx - half_size)
       y1 = max(0, cy - half_size)
       x2 = min(w, x1 + current_crop_size)
       y2 = min(h, y1 + current_crop_size)
       img_cropped = img_processed.crop((x1, y1, x2, y2))
   else:
       # 没脸就居中裁个正方形
       dim = min(w, h)
       l, t = (w-dim)//2, (h-dim)//2
       img_cropped = img_processed.crop((l, t, l+dim, t+dim))
   # 显示预览图
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="裁剪预览", use_container_width=True)
   # 只有点击按钮才计算重型逻辑
   if submit_btn:
       with st.spinner("正在优先分配人脸积木..."):
           # 缩放
           img_small = img_cropped.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
           pixel_data = np.array(img_small)
           # 生成人脸 Mask
           mask = generate_priority_mask(img_cropped, grid_size)
           # 运行核心分配逻辑
           final_canvas, usage = process_priority_allocation(
               pixel_data, mask, LEGO_31205_DATA, use_dithering_bg
           )
           res_img = Image.fromarray(final_canvas.astype('uint8'))
           with col2:
               st.image(res_img.resize((600, 600), Image.Resampling.NEAREST),
                        caption="最终效果", use_container_width=True)
           # 库存预警可视化
           st.write("---")
           st.subheader("📊 零件消耗情况")
           # 将字典转为列表排序
           usage_list = []
           for k, v in LEGO_31205_DATA.items():
               used = usage.get(k, 0)
               remaining = v[1] - used
               status = "✅ 充足"
               if remaining < 0: status = "❌ 缺件 (逻辑错误)" # 理论上不会出现
               elif remaining == 0: status = "⚠️ 耗尽"
               elif remaining < 50: status = "📉 紧张"
               usage_list.append({
                   "颜色": k,
                   "已用": used,
                   "剩余": remaining,
                   "状态": status
               })
           st.dataframe(usage_list, use_container_width=True)
