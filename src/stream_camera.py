import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "192.168.1.2"
PORT = 8080

CAMERA_ID = 0
WIDTH = 640
HEIGHT = 480
FPS = 15
JPEG_QUALITY = 70

camera = cv2.VideoCapture(CAMERA_ID)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

if not camera.isOpened():
    raise RuntimeError("Could not open laptop camera")


class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path != "/video":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Use /video")
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        try:
            while True:
                ok, frame = camera.read()

                if not ok:
                    break

                frame = cv2.resize(frame, (WIDTH, HEIGHT))

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )

                if not ok:
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(encoded)}\r\n\r\n".encode()
                )
                self.wfile.write(encoded.tobytes())
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - {format % args}")


server = ThreadingHTTPServer((HOST, PORT), StreamHandler)

print("========================================")
print("Laptop Camera MJPEG Stream")
print(f"Camera: /dev/video{CAMERA_ID}")
print(f"Stream: http://{HOST}:{PORT}/video")
print("Press Ctrl+C to stop.")
print("========================================")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nStopping stream...")
finally:
    camera.release()
    server.server_close()
