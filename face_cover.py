import cv2
import numpy as np
cover_img = cv2.imread("./img/face_img.png", cv2.IMREAD_UNCHANGED)

def overlay_image_alpha(img, x, y, overlay_size=None):
    print("start overlay")
    if cover_img is None:
        return img
    overlay_img = cover_img.copy()
    if overlay_size is not None:
        overlay_img = cv2.resize(overlay_img, overlay_size)
    # カバー画像の中心を外接矩形の中心に合わせる
    h, w = overlay_img.shape[:2]
    x -= w // 2
    y -= h // 2

    # はみ出しを調整
    x_end = min(x + w, img.shape[1])
    y_end = min(y + h, img.shape[0])
    x, y = max(x, 0), max(y, 0)

    # 重ね合わせる範囲を調整
    overlay_width = x_end - x
    overlay_height = y_end - y
    if overlay_width <= 0 or overlay_height <= 0:
        return img  # 重ね合わせが不要
    # 画像の一部を切り取る
    # アルファチャンネルの確認
    if overlay_img.shape[2] == 4:
        # アルファチャンネルがある場合
        b, g, r, a = cv2.split(overlay_img)
        overlay_color = cv2.merge((b, g, r))
        mask = a / 255.0
        mask = np.stack((mask,) * 3, axis=-1)
    else:
        # アルファチャンネルがない場合
        overlay_color = overlay_img
        mask = np.ones_like(overlay_color, dtype=np.float32)

    # 元の画像の該当部分を取得
    img[y:y+overlay_height, x:x+overlay_width] = (
            1.0 - mask[:overlay_height, :overlay_width]) * img[y:y+overlay_height, x:x+overlay_width] + \
            mask[:overlay_height, :overlay_width] * overlay_color[:overlay_height, :overlay_width]
    return img