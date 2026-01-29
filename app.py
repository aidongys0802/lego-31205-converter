import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
# --- 1. 基础配置与乐高数据 ---
st.set_page_config(page_title="LEGO 31205 定制优化版", layout="wide")
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
# --- 2. 核心算法组件 ---
def get_closest_color_info(target_rgb, inventory):
   """找到最接近的颜色"""
   tr, tg, tb = target_rgb
   best_dist = float('inf')
   best_key = "Black"
   # 只查找，不扣库存
   for name, data in inventory.items():
       (r, g, b), count = data
       if count > 0:
           dist = (r - tr)**2 + (g - tg)**2 + (b - tb)**2
           if dist < best_dist:
               best_dist = dist
               best_key = name
   return inventory[best_key][0], best_key
def detect_face_rect(pil_img):
   """在图片上检测最大人脸的矩形区域"""
   face_cascade = load_face_cascade()
   cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
   gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   if len(faces) > 0:
       # 返回最大的脸
       return sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
   return None
def smart_crop_by_rect(pil_img, face_rect, zoom_level=1.5):
   """基于给定的人脸矩形进行智能裁剪"""
   w, h = pil_img.size
   if face_rect is not None:
       fx, fy, fw, fh = face_rect
       cx, cy = fx + fw // 2, fy + fh // 2
       crop_dim = int(max(fw, fh) * zoom_level)
       x1 = max(0, cx - crop_dim // 2)
       y1 = max(0, cy - crop_dim // 2)
       x2 = min(w, cx + crop_dim // 2)
       y2 = min(h, cy + crop_dim // 2)
       final_dim = min(x2 - x1, y2 - y1)
       return pil_img.crop((x1, y1, x1+final_dim, y1+final_dim))
   else:
       dim = min(w, h)
       left, top = (w - dim) // 2, (h - dim) // 2
       return pil_img.crop((left, top, left + dim, top + dim))
def generate_face_mask_lowres(pil_img_cropped, grid_size):
   """生成一个低分辨率的 Mask，白色表示人脸核心区域"""
   # 1. 在高分辨率裁剪图上再次定位人脸
   face_rect = detect_face_rect(pil_img_cropped)
   # 创建全黑高分 Mask
   mask_hr_np = np.zeros((pil_img_cropped.size[1], pil_img_cropped.size[0]), dtype=np.uint8)
   if face_rect is not None:
       fx, fy, fw, fh = face_rect
       # 稍微向内收缩一点，只保护核心五官区域不抖动，边缘可以稍微过渡
       inset_x = int(fw * 0.15)
       inset_y = int(fh * 0.1)
       # 在 Mask 上绘制白色实心矩形
       cv2.rectangle(mask_hr_np, (fx+inset_x, fy+inset_y), (fx+fw-inset_x, fy+fh-inset_y), 255, -1)
   mask_hr_img = Image.fromarray(mask_hr_np)
   # 2. 使用最近邻插值缩放到网格尺寸，保证边缘锐利
   mask_lr_img = mask_hr_img.resize((grid_size, grid_size), Image.Resampling.NEAREST)
   return np.array(mask_lr_img)
def apply_selective_dithering(pixel_array, face_mask_array, inventory, use_dithering_bg=True):
   """应用选择性抖动：人脸区域强制平滑，背景可选抖动"""
   h, w, _ = pixel_array.shape
   buffer = pixel_array.astype(float)
   output = np.zeros_like(pixel_array)
   stats = {}
   temp_inv = {k: [list(v[0]), v[1]] for k, v in inventory.items()}
   for y in range(h):
       for x in range(w):
           old_pixel = buffer[y, x]
           # 如果 mask 对应位置是白色 (255)，则是人脸保护区
           is_face_protected = face_mask_array[y, x] > 128
           new_pixel, name = get_closest_color_info(old_pixel, temp_inv)
           output[y, x] = new_pixel
           stats[name] = stats.get(name, 0) + 1
           temp_inv[name][1] -= 1
           # 关键逻辑：只有当 (开启了背景抖动) 且 (当前像素不在人脸保护区) 时，才扩散误差
           if use_dithering_bg and not is_face_protected:
               quant_error = old_pixel - new_pixel
               if x + 1 < w: buffer[y, x + 1] += quant_error * 7 / 16
               if y + 1 < h:
                   if x - 1 >= 0: buffer[y + 1, x - 1] += quant_error * 3 / 16
                   buffer[y + 1, x] += quant_error * 5 / 16
                   if x + 1 < w: buffer[y + 1, x + 1] += quant_error * 1 / 16
   return output, stats
# --- 3. 界面主逻辑 ---
st.title("🧩 LEGO 31205 人像定制优化版")
st.markdown("**优化重点：人脸区域无抖动、肤色统一、五官清晰。**")
with st.sidebar:
   st.header("🎛️ 参数面板")
   grid_size = st.select_slider("画布分辨率 (Grid Size)", options=[32, 48, 64], value=48)
   st.subheader("1. 构图与预处理")
   zoom_factor = st.slider("人脸特写程度 (数值越小脸越大)", 1.3, 3.0, 2.0)
   contrast = st.slider("对比度增强 (提高清晰度)", 0.8, 1.8, 1.3, help="增加对比度有助于让五官与肤色分离得更清晰")
   brightness = st.slider("亮度调整 (使肤色更浅)", 0.8, 1.5, 1.1, help="适当提高亮度可以让肤色匹配到更浅的积木")
   st.subheader("2. 质感控制")
   use_dithering_bg = st.checkbox("背景开启抖动质感", value=True, help="人脸区域将始终保持光滑统一，此选项仅影响背景和衣服。")
uploaded_file = st.file_uploader("上传照片 (建议面部光线均匀)", type=["jpg", "png", "jpeg"])
if uploaded_file:
   # 1. 加载与预处理
   original = Image.open(uploaded_file).convert("RGB")
   # 调整亮度和对比度
   enhancer_bri = ImageEnhance.Brightness(original)
   img_bri = enhancer_bri.enhance(brightness)
   enhancer_con = ImageEnhance.Contrast(img_bri)
   img_processed = enhancer_con.enhance(contrast)
   # 2. 智能检测与裁剪
   face_rect_raw = detect_face_rect(img_processed)
   img_cropped = smart_crop_by_rect(img_processed, face_rect_raw, zoom_level=zoom_factor)
   col1, col2 = st.columns(2)
   with col1:
       st.image(img_cropped, caption="预处理与裁剪结果", use_container_width=True)
   if st.button("生成定制乐高画"):
       with st.spinner("正在进行分区纹理处理..."):
           # 3. 生成人脸保护 Mask (低分辨率)
           face_mask_lr = generate_face_mask_lowres(img_cropped, grid_size)
           # Debug: 取消下面注释可以预览人脸保护区域
           # st.image(Image.fromarray(face_mask_lr), caption="人脸保护区预览(白色区域不抖动)", width=200)
           # 4. 缩放图像并应用选择性抖动
           img_small = img_cropped.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
           pixel_data = np.array(img_small)
           final_array, usage = apply_selective_dithering(
               pixel_data, face_mask_lr, LEGO_31205_DATA, use_dithering_bg
           )
           result_img = Image.fromarray(final_array.astype('uint8'))
           with col2:
               st.image(result_img.resize((600, 600), Image.Resampling.NEAREST),
                        caption="最终效果 (人脸光滑优化)", use_container_width=True)
           st.success("生成完成！脸部区域已自动净化噪点。")
           with st.expander("查看零件消耗清单"):
               sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
               st.table([{"零件颜色": k, "使用数量": v, "库存剩余": LEGO_31205_DATA[k][1]-v} for k, v in sorted_usage])
