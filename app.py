import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.solutions import selfie_segmentation as mp_selfie_seg
from mediapipe.solutions import face_detection as mp_face_det
from PIL import Image
import math
# --- 1. 配置与乐高 31205 数据 ---
st.set_page_config(page_title="LEGO 31205 人像转换器", layout="wide")
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
# 初始化模型 (显式导入子模块以增强兼容性)
@st.cache_resource
def load_models():
   mp_selfie = mp_selfie_seg.SelfieSegmentation(model_selection=1)
   mp_face = mp_face_det.FaceDetection(model_selection=1, min_detection_confidence=0.5)
   return mp_selfie, mp_face
# --- 2. 核心算法逻辑 ---
def get_closest_available(target_rgb, inventory):
   r, g, b = target_rgb
   candidates = []
   for name, data in inventory.items():
       rgb, count = data
       if count > 0:
           # 简单的欧式距离计算颜色接近度
           dist = math.sqrt((r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2)
           candidates.append((dist, name))
   if not candidates:
       return (0, 0, 0), "Black"
   candidates.sort()
   best_name = candidates[0][1]
   inventory[best_name][1] -= 1  # 消耗一个零件
   return inventory[best_name][0], best_name
def process_image(pil_img, size, p_weights):
   mp_selfie, mp_face = load_models()
   # 强制转换为 RGB Numpy 数组
   img_rgb = np.array(pil_img.convert("RGB"))
   if img_rgb is None or img_rgb.size == 0:
       return None, None
   h, w, _ = img_rgb.shape
   # 居中裁剪成正方形
   crop_size = min(h, w)
   y0, x0 = (h - crop_size)//2, (w - crop_size)//2
   cropped_img = img_rgb[y0:y0+crop_size, x0:x0+crop_size]
   # AI 分析 (使用原图或裁剪图)
   res_seg = mp_selfie.process(cropped_img)
   person_mask = res_seg.segmentation_mask # 0-1 之间的概率图
   res_face = mp_face.process(cropped_img)
   face_boxes = []
   if res_face.detections:
       for det in res_face.detections:
           bbox = det.location_data.relative_bounding_box
           face_boxes.append(bbox)
   # 缩放到颗粒度大小
   img_s = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_AREA)
   mask_s = cv2.resize(person_mask, (size, size), interpolation=cv2.INTER_NEAREST)
   if img_s is None or img_s.size == 0:
       return None, None
   img_hsv = cv2.cvtColor(img_s, cv2.COLOR_RGB2HSV)
   # 优先级评分
   pixel_tasks = []
   for y in range(size):
       for x in range(size):
           rel_x, rel_y = x / size, y / size
           is_person = mask_s[y, x] > 0.5
           is_face = False
           if is_person:
               for box in face_boxes:
                   if (box.xmin <= rel_x <= box.xmin + box.width and
                       box.ymin <= rel_y <= box.ymin + box.height):
                       is_face = True
                       break
           v_val = img_hsv[y, x, 2] # 亮度
           if is_face: score = p_weights['face']
           elif is_person: score = p_weights['clothes']
           elif v_val > 200: score = p_weights['bg_high']
           elif v_val < 50: score = p_weights['bg_dark']
           else: score = p_weights['bg_normal']
           pixel_tasks.append({'pos':(x,y), 'rgb':img_s[y,x], 'score':score})
   # 根据优先级排序，优先分配重要部位的零件颜色
   pixel_tasks.sort(key=lambda t: t['score'], reverse=True)
   # 拷贝一份库存进行计算
   curr_inv = {k: [v[0], v[1]] for k, v in LEGO_31205_DATA.items()}
   res_pixels = {}
   usage = {}
   for task in pixel_tasks:
       rgb, name = get_closest_available(task['rgb'], curr_inv)
       res_pixels[task['pos']] = rgb
       usage[name] = usage.get(name, 0) + 1
   # 生成预览图
   out_img = Image.new("RGB", (size, size))
   pix = out_img.load()
   for pos, rgb in res_pixels.items():
       pix[pos[0], pos[1]] = tuple(map(int, rgb))
   return out_img, usage
# --- 3. 网页界面布局 ---
st.title("🧩 LEGO 31205 艺术画智能生成器")
st.markdown("上传照片，AI 将自动识别人物并根据积木库存优化分配颜色。")
with st.sidebar:
   st.header("⚙️ 参数设置")
   grid_size = st.slider("画布尺寸 (颗粒数)", 16, 128, 48)
   st.subheader("优先级权重自定")
   w_face = st.number_input("人物面部", value=2000)
   w_clothes = st.number_input("人物衣着", value=1000)
   w_high = st.number_input("背景高光", value=500)
   w_normal = st.number_input("背景普通", value=200)
   w_dark = st.number_input("背景阴影", value=100)
   st.info("权重越高，该区域越优先匹配最接近的颜色。")
uploaded_file = st.file_uploader("选择一张照片...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
   image = Image.open(uploaded_file)
   col1, col2 = st.columns(2)
   with col1:
       st.image(image, caption="原始照片", use_container_width=True)
   if st.button("开始生成乐高预览"):
       with st.spinner('AI 正在分析人像并计算积木分配...'):
           p_weights = {
               'face': w_face, 'clothes': w_clothes,
               'bg_high': w_high, 'bg_normal': w_normal, 'bg_dark': w_dark
           }
           result_img, usage_stats = process_image(image, grid_size, p_weights)
           if result_img:
               with col2:
                   # 使用 Nearest Neighbor 放大，保持像素感
                   st.image(result_img.resize((600, 600), resample=0),
                            caption="乐高效果预览", use_container_width=True)
               st.success("生成成功！")
               # 显示零件消耗统计
               st.subheader("📊 零件消耗统计")
               cols = st.columns(3)
               for i, (name, count) in enumerate(usage_stats.items()):
                   original_stock = LEGO_31205_DATA[name][1]
                   cols[i % 3].metric(name, f"{count} 颗", f"剩余 {original_stock - count}")
           else:
               st.error("处理失败，请检查图片格式。")
