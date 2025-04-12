import cv2

def use_specific_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Nem sikerült megnyitni a kamerát index: {camera_index}")
        return

    print(f"Kamera megnyitva index: {camera_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nem sikerült képkockát olvasni.")
            break

        cv2.imshow("Kamera kiválasztása", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    use_specific_camera(0)  # Cseréld le a számítógép kamerájának indexére
