import os
import sys
# 現在のディレクトリにあるARディレクトリへの絶対パスを取得
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv

VISUALIZE_FACE_POINTS = False



filters_config = {
    'anonymous':
        [{'path': "filters/anonymous.png",
          'anno_path': "filters/anonymous.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anime':
        [{'path': "filters/anime.png",
          'anno_path': "filters/anime.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "filters/dog-ears.png",
          'anno_path': "filters/dog-ears.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/dog-nose.png",
          'anno_path': "filters/dog-nose.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "filters/cat-ears.png",
          'anno_path': "filters/cat-ears.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/cat-nose.png",
          'anno_path': "filters/cat-nose.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "filters/jason-joker.png",
          'anno_path': "filters/jason-joker.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'gold-crown':
        [{'path': "filters/gold-crown.png",
          'anno_path': "filters/gold-crown.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "filters/flower-crown.png",
          'anno_path': "filters/flower-crown.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'smile':
        [{'path': "filters/smile.png",
          'anno_path': "filters/smile.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'smily':
        [{'path': "filters/smily.png",
          'anno_path': "filters/smily.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
}


# detect facial landmarks in image
def getLandmarks(img, min_detection_confidence=0.5, max_num_faces=10):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=max_num_faces, static_image_mode=True, min_detection_confidence=min_detection_confidence) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return []

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return []


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        print(os.path.join(current_dir, filter['path']))
        img1, img1_alpha = load_filter_img(os.path.join(current_dir, filter['path']), filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(os.path.join(current_dir, filter['anno_path']))
        print("load points", len(points))

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime


def apply_face_mask(frame, mask_name, points2):
    """
    顔にフィルターをかけるメイン処理
    :param frame: ビデオフレーム
    :param mask_name: 使用するフィルター名
    :param points2: フィルター適用対象の顔の特徴点
    :return: マスクが適用されたフレーム
    """
    # 顔が検出されなかった場合の処理
    if not points2 or len(points2) != 75:
        return frame

    filters, multi_filter_runtime = load_filter(mask_name)  # 使用するフィルターを指定

    for idx, filter in enumerate(filters):
        filter_runtime = multi_filter_runtime[idx]
        img1 = filter_runtime['img']
        points1 = filter_runtime['points']
        img1_alpha = filter_runtime['img_a']

        # フィルターが顔に合わせて変形する場合
        if filter['morph']:
            hullIndex = filter_runtime['hullIndex']
            dt = filter_runtime['dt']
            hull1 = filter_runtime['hull']
            warped_img = np.copy(frame)
            hull2 = [points2[hullIndex[i][0]] for i in range(len(hullIndex))]

            mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
            mask1 = cv2.merge((mask1, mask1, mask1))
            img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

            # 各三角形領域を変形して重ね合わせ
            for i in range(len(dt)):
                t1 = [hull1[dt[i][j]] for j in range(3)]
                t2 = [hull2[dt[i][j]] for j in range(3)]

                if is_valid_triangle(t1, warped_img.shape) and is_valid_triangle(t2, warped_img.shape):
                    fbc.warpTriangle(img1, warped_img, t1, t2)
                    fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

            # マスクをぼかして合成
            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
            mask2 = (255.0, 255.0, 255.0) - mask1
            temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
            frame = np.uint8(temp1 + temp2)

        else:
            # 顔に合わせてフィルターを移動して合成
            try:
                dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                tform = fbc.similarityTransform(list(points1.values()), dst_points)

                trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))

                mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                # マスク領域を画面サイズ内にクリップ
                mask1 = np.clip(mask1, 0, 255)

                mask2 = (255.0, 255.0, 255.0) - mask1
                temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                frame = np.uint8(temp1 + temp2)

            except Exception as e:
                print(f"Error applying filter: {e}")
                continue

    return frame


def is_valid_triangle(triangle, img_shape):
    """
    三角形が画像範囲内かどうかを判定する
    :param triangle: 三角形を構成する3点
    :param img_shape: 画像の形状
    :return: True if valid, False otherwise
    """
    h, w = img_shape[:2]
    for x, y in triangle:
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
    return True



if __name__=="__main":
    pass