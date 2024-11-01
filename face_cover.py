import cv2
cover_img = cv2.imread("./img/face_img.png")

def overlay_image_alpha(img, x, y, overlay_size=None):
    """アルファチャンネル付きの画像を、範囲内に収まるように重ね合わせる"""
    if overlay_size is not None:
        cover_img = cv2.resize(cover_img, overlay_size)

    # カバー画像の中心を外接矩形の中心に合わせる
    h, w = cover_img.shape[:2]
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
    b, g, r, a = cv2.split(cover_img)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0

    # 元の画像に上書き
    img[y:y+overlay_height, x:x+overlay_width] = (
        1.0 - mask[:overlay_height, :overlay_width]) * img[y:y+overlay_height, x:x+overlay_width] + \
        mask[:overlay_height, :overlay_width] * overlay_color[:overlay_height, :overlay_width]
    return img