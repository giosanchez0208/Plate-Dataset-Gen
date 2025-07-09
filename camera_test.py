import cv2
import torch
import argparse
from ultralytics import YOLO

def run_on_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    results = model(image)
    annotated_image = results[0].plot()
    cv2.imshow('Segmentation Result', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_webcam(model):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow('Segmentation Test', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to an image file to run segmentation on.')
    args = parser.parse_args()

    model = YOLO('best.pt')

    if args.image:
        run_on_image(args.image, model)
    else:
        run_on_webcam(model)

if __name__ == "__main__":
    main()
