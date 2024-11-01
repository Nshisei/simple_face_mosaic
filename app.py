import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import os

from face_cover import overlay_image_alpha

def face_detection_using_mediapipe():
    min_detection_confidence = st.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    def callback(frame):
        mp_face_detection = mp.solutions.face_detection
        image = frame.to_ndarray(format="bgr24")
        with mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as face_detection:
            # 画像の左右反転
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = face_detection.process(image)
            
            # 顔検出された場合、各顔にカバー画像を適用
            if results.detections:
                for detection in results.detections:
                    # 顔の外接矩形を取得
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    box_width = int(bbox.width * w)
                    box_height = int(bbox.height * h)
                    
                    # カバー画像を重ねる
                    overlay_size = (int(box_width * 1.5), int(box_height * 1.5))
                    image = overlay_image_alpha(image, x_min + box_width // 2, y_min + box_height // 2, overlay_size)
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(key="face-overlay-mediapipe", video_frame_callback=callback, async_processing=True, media_stream_constraints={'video': True, 'audio': False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

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
                                          'Face Detection using Opencv', ))
    
    func_dict = {
    'Face Detection using Mediapipe': face_detection_using_mediapipe,
    'Face Detection using Opencv': face_detection_using_opencv,
    }

    func_dict.get(select_app, lambda: st.error('Invalid Selection'))()    


if __name__ == '__main__':
    main()