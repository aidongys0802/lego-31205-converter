import streamlit as st

import numpy as np

import cv2

from PIL import Image, ImageEnhance

import random

st.set_page_config(page_title="LEGO 31205 艺术大师版", layout="wide")

# 1. 零件库

def get_lego_inventory():

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

@st.cache_resource

def load_cascade():

    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 增强型颜色匹配 (支持皮肤优化逻辑)

def find_color_logic(target_rgb, inv, is_skin=False, sparse_dither=0.1):

    tr, tg, tb = target_rgb

    # 如果是皮肤区域且触发了随机稀疏抖动

    if is_skin and random.random() < sparse_dither:

        # 在皮肤中加入少量白色高光或浅色过渡

        target_rgb = [min(255, tr+20), min(255, tg+20), min(255, tb+20)]

        tr, tg, tb = target_rgb

    best_dist = float('inf')

    best_key = "Black"

    for name, data in inv.items():

        (r, g, b), count = data

        if count > 0:

            # 皮肤区域限制：只允许 Flesh, Tan, White, Dark Orange

            if is_skin and name not in ["Flesh", "Tan", "White", "Dark Orange"]:

                continue

            dist = 2*(r-tr)**2 + 4*(g-tg)**2 + 3*(b-tb)**2

            if dist < best_dist:

                best_dist, best_key = dist, name

    inv[best_key][1] -= 1

    return inv[best_key][0]

# 3. 侧边栏控制

st.sidebar.header("🛠️ 模式与优化")

performance_mode = st.sidebar.toggle("性能模式 (预览更快)", value=False)

st.sidebar.divider()

st.sidebar.header("🎨 皮肤特调")

skin_dither_val = st.sidebar.slider("皮肤抖动稀疏度", 0.0, 0.3, 0.05, step=0.01)

skin_brightness = st.sidebar.slider("皮肤亮度", 0.5, 2.0, 1.1)

st.sidebar.header("📏 基础设置")

grid_size = st.sidebar.select_slider("分辨率", options=[32, 48, 64], value=48)

zoom = st.sidebar.slider("对焦", 1.0, 3.0, 2.0)

contrast = st.sidebar.slider("五官对比度", 0.5, 2.5, 1.4)

uploaded_file = st.file_uploader("上传人像", type=["jpg", "png", "jpeg"])

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    # 裁剪与对焦

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    faces = load_cascade().detectMultiScale(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), 1.1, 4)

    w, h = img.size

    if len(faces) > 0:

        fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]

        cx, cy = fx + fw//2, fy + fh//2

        dim = int(min(w, h) / zoom)

        img_cropped = img.crop((max(0, cx-dim//2), max(0, cy-dim//2), min(w, cx+dim//2), min(h, cy+dim//2)))

    else:

        dim = min(w, h)

        img_cropped = img.crop(((w-dim)//2, (h-dim)//2, (w+dim)//2, (h+dim)//2))

    # 增强处理

    processed = ImageEnhance.Contrast(img_cropped).enhance(contrast)

    processed = ImageEnhance.Brightness(processed).enhance(skin_brightness)

    col1, col2 = st.columns(2)

    with col1:

        st.image(img_cropped, caption="对焦预览", use_container_width=True)

    # 4. 核心运算

    if not performance_mode:

        with st.spinner("深度优化皮肤与五官中..."):

            small = processed.resize((grid_size, grid_size), Image.Resampling.LANCZOS)

            pixels = np.array(small)

            # 皮肤检测 (基于 HSV 范围)

            hsv_small = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)

            # 定义大致肤色范围

            lower_skin = np.array([0, 20, 70], dtype=np.uint8)

            upper_skin = np.array([25, 255, 255], dtype=np.uint8)

            skin_mask = cv2.inRange(hsv_small, lower_skin, upper_skin)

            inv = {k: [list(v[0]), v[1]] for k, v in get_lego_inventory().items()}

            canvas = np.zeros_like(pixels)

            # 优先级分配

            # 第一遍：处理皮肤 (应用受控抖动)

            for y in range(grid_size):

                for x in range(grid_size):

                    if skin_mask[y, x] > 0:

                        canvas[y, x] = find_color_logic(pixels[y, x], inv, is_skin=True, sparse_dither=skin_dither_val)

            # 第二遍：处理五官及背景

            for y in range(grid_size):

                for x in range(grid_size):

                    if skin_mask[y, x] == 0:

                        canvas[y, x] = find_color_logic(pixels[y, x], inv, is_skin=False)

            with col2:

                res_img = Image.fromarray(canvas.astype('uint8'))

                st.image(res_img.resize((600, 600), Image.Resampling.NEAREST), caption="深度优化效果", use_container_width=True)

    else:

        # 性能模式：简化计算

        small = processed.resize((grid_size, grid_size), Image.Resampling.NEAREST)

        with col2:

            st.image(small.resize((600, 600), Image.Resampling.NEAREST), caption="性能预览模式 (非最终效果)", use_container_width=True)

            st.warning("性能模式开启中，已禁用高级皮肤优化。")
 
