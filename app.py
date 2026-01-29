import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math
# 乐高 31205 官方数据
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
# 初始化 AI 模型
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
def get_closest_available(target_rgb, inventory):
   r, g, b = target_rgb
   candidates = []
   for name, data in inventory.items():
       rgb, count = data
       if count > 0:
           dist = math.sqrt((r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2)
           candidates.append((dist, name))
   if not candidates: return (0,0,0), "Black"
   candidates.sort()
   best_name = candidates[0][1]
   inventory[best_name][1] -= 1
   return inventory[best_name][0], best_name
def convert_with_strict_priority(image_path, size=48):
   # 1. 读取并预处理
   img_bgr = cv2.imread(image_path)
   img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
   h, w, _ = img_rgb.shape
   # 2. AI 分析
   # 人像分割
   res_seg = mp_selfie.process(img_rgb)
   person_mask = res_seg.segmentation_mask # 0-1 概率
   # 人脸检测
   res_face = mp_face.process(img_rgb)
   face_boxes = []
   if res_face.detections:
       for det in res_face.detections:
           bbox = det.location_data.relative_bounding_box
           face_boxes.append(bbox) # 存储相对坐标
   # 3. 裁剪并缩放到 48x48
   crop_size = min(h, w)
   y0, x0 = (h - crop_size)//2, (w - crop_size)//2
   img_s = cv2.resize(img_rgb[y0:y0+crop_size, x0:x0+crop_size], (size, size))
   mask_s = cv2.resize(person_mask[y0:y0+crop_size, x0:x0+crop_size], (size, size))
   img_hsv = cv2.cvtColor(img_s, cv2.COLOR_RGB2HSV) # 转 HSV 用于分析亮度
   # 4. 优先级评分逻辑
   pixel_tasks = []
   for y in range(size):
       for x in range(size):
           rel_x, rel_y = x / size, y / size
           is_person = mask_s[y, x] > 0.5
           # 检查是否在人脸框内
           is_face = False
           if is_person:
               for box in face_boxes:
                   # 简化逻辑：将相对坐标映射到裁剪后的区域
                   if (box.xmin <= rel_x <= box.xmin + box.width and
                       box.ymin <= rel_y <= box.ymin + box.height):
                       is_face = True
                       break
           # 计算背景亮度 (V 通道)
           v_value = img_hsv[y, x, 2]
           # 分配分数
           if is_face:
               score = 1000
           elif is_person:
               score = 800
           elif v_value > 200: # 背景高光
               score = 600
           elif v_value < 50:  # 背景阴影
               score = 200
           else:               # 背景其他
               score = 400
           pixel_tasks.append({'pos':(x,y), 'rgb':img_s[y,x], 'score':score})
   # 5. 按分数排序并分配库存
   pixel_tasks.sort(key=lambda t: t['score'], reverse=True)
   inventory = {k: [v[0], v[1]] for k, v in LEGO_31205_DATA.items()}
   res_pixels = {}
   for task in pixel_tasks:
       rgb, _ = get_closest_available(task['rgb'], inventory)
       res_pixels[task['pos']] = rgb
   # 6. 渲染输出
   out_img = Image.new("RGB", (size, size))
   pix = out_img.load()
   for pos, rgb in res_pixels.items():
       pix[pos[0], pos[1]] = rgb
   final = out_img.resize((480, 480), Image.Resampling.NEAREST)
   final.save("priority_result.png")
   print("转换完成！已根据 面部 > 衣着 > 背景高光 > 其他 > 阴影 的顺序分配积木。")
   final.show()
if __name__ == "__main__":
   convert_with_strict_priority("photo.jpg")