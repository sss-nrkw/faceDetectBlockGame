from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

    ball_radius = 10
    ball_x, ball_y = 320, 240  # 初期位置を中央に設定
    ball_dx, ball_dy = 3, 3  # 移動量を設定
    

    bar_width, bar_height = 150, 15
    bar_x = (640 - bar_width) // 2
    bar_y = 480 - 2 * bar_height

    running = True
    game_over = False

    while running:
        ret, frame = cap.read()

        if not game_over:
            ball_x += ball_dx
            ball_y += ball_dy

            if ball_x <= 0 or ball_x >= 640:
                ball_dx *= -1
            if ball_y <= 0:
                ball_dy *= -1

            if ball_y >= bar_y - ball_radius and bar_x <= ball_x <= bar_x + bar_width:
                ball_dy *= -1
            elif ball_y >= 480:
                game_over = True

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            bar_x = x  # 顔の X 座標をバーの X 座標に使用

        cv2.circle(frame, (int(ball_x), int(ball_y)), ball_radius, (255, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), -1)

        if game_over:
            cv2.putText(frame, "Game Over!", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if game_over:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
