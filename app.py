import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import os

from face_cover import overlay_image_alpha
import os
import sys
current_dir = os.path.dirname(__file__)
ar_dir = os.path.join(current_dir, 'AR')
sys.path.append(ar_dir)
from apply_filter import apply_face_mask, getLandmarks

def face_detection_using_mediapipe():
    min_detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    def callback(frame):
        mp_face_detection = mp.solutions.face_detection
        image = frame.to_ndarray(format="bgr24")
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            if results.detections:
                for detection in results.detections:
                    
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    box_width = int(bbox.width * w)
                    box_height = int(bbox.height * h)

                    # 検出の信頼度を取得して表示
                    confidence = detection.score[0]
                    if min_detection_confidence > confidence:
                        break
                    text = f'{confidence:.2f}'
                    cv2.putText(image, text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


                    # カバー画像を重ねる
                    overlay_size = (int(box_width * 1.5), int(box_height * 1.5))
                    print("detection process")
                    image = overlay_image_alpha(image, x_min + box_width // 2, y_min + box_height // 2, overlay_size)
                    print("end overlay")
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(key="face-overlay-mediapipe", video_frame_callback=callback, async_processing=True, media_stream_constraints={'video': True, 'audio': False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 顔検出とFaceMeshを使用した関数
def face_mesh_using_mediapipe():
    min_detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    def callback(frame):
        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), min_detection_confidence=0.3)
        image = apply_face_mask(frame, points2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(
        key="facemesh-overlay-mediapipe",
        video_frame_callback=callback,
        async_processing=True,
        media_stream_constraints={'video': True, 'audio': False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

def face_detection_using_opencv():

    def callback(frame):
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        face_cascade_profile = cv2.CascadeClassifier('./haarcascade_profileface.xml')

        image = frame.to_ndarray(format="bgr24")
        src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        def combine_rects(rects):
            """複数の矩形領域を結合して外接矩形を作成する"""
            if len(rects) == 0:
                return None
            x_min = min([x for (x, y, w, h) in rects])
            y_min = min([y for (x, y, w, h) in rects])
            x_max = max([x + w for (x, y, w, h) in rects])
            y_max = max([y + h for (x, y, w, h) in rects])
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # 顔検出
        faces1 = face_cascade.detectMultiScale(src_gray)
        faces2 = face_cascade_profile.detectMultiScale(src_gray)

        # 画像を左右反転して横顔の検出を試みる
        flipped_gray = cv2.flip(src_gray, 1)
        faces3 = face_cascade_profile.detectMultiScale(flipped_gray)

        # 反転した座標を元に戻す
        frame_width = frame.shape[1]
        faces3 = [(frame_width - x - w, y, w, h) for (x, y, w, h) in faces3]

        # すべての顔領域を結合
        all_faces = list(faces1) + list(faces2) + faces3

        # 重なっている領域を1つにまとめる
        combined_face_rect = combine_rects(all_faces)
        if combined_face_rect:
            x, y, w, h = combined_face_rect
            cx, cy = x + w // 2, y + h // 2  # 中心座標を計算

            # カバー画像を重ねる
            frame = overlay_image_alpha(frame, cx, cy, overlay_size=(w,h))
                    
            return av.VideoFrame.from_ndarray(image, format="bgr24")


    webrtc_streamer(key="face-overlay-opencv", video_frame_callback=callback, async_processing=True, media_stream_constraints={'video': True, 'audio': False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def main():
    st.title('Moke up app')

    select_app = st.sidebar.radio('app', ('Face Detection using Mediapipe', 
                                          'Face Mesh using Mediapipe',
                                          'Face Detection using Opencv', ))
    
    func_dict = {
    'Face Detection using Mediapipe': face_detection_using_mediapipe,
    'Face Mesh using Mediapipe': face_mesh_using_mediapipe,
    'Face Detection using Opencv': face_detection_using_opencv,
    }

    func_dict.get(select_app, lambda: st.error('Invalid Selection'))()    


if __name__ == '__main__':
    main()