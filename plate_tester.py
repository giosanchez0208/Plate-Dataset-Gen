import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    UPSCALE_AVAILABLE = True
except ImportError:
    UPSCALE_AVAILABLE = False

def smart_resize(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    padded = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pad_h = (target_size - resized.shape[0]) // 2
    pad_w = (target_size - resized.shape[1]) // 2
    padded[pad_h:pad_h + resized.shape[0], pad_w:pad_w + resized.shape[1]] = resized
    return padded

def upscale_image(image, upscaler):
    if upscaler is None:
        return image
    try:
        result, _ = upscaler.enhance(image, outscale=2)
        return result
    except:
        return image

def get_line_groups(boxes, y_tolerance=0.3):
    if len(boxes) == 0:
        return []
    
    boxes_with_centers = [(box, (box[1] + box[3]) / 2) for box in boxes]
    boxes_with_centers.sort(key=lambda x: x[1])
    
    groups = []
    current_group = [boxes_with_centers[0]]
    
    for i in range(1, len(boxes_with_centers)):
        box, y_center = boxes_with_centers[i]
        prev_y = current_group[-1][1]
        
        box_height = box[3] - box[1]
        if abs(y_center - prev_y) <= box_height * y_tolerance:
            current_group.append((box, y_center))
        else:
            groups.append(current_group)
            current_group = [(box, y_center)]
    
    groups.append(current_group)
    return groups

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - interArea
    return interArea / denom if denom > 0 else 0

def nms(boxes, scores, thresh=0.3):
    keep = []
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < thresh]
    return keep

def main():
    upscaler = None
    if UPSCALE_AVAILABLE:
        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            upscaler = RealESRGANer(scale=2, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth', model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
        except:
            pass
    
    plate_model = YOLO('best_license_plate.pt')
    text_model = YOLO('best_text_reco.pt')
    paths = sorted([p for p in glob.glob('example/*') if os.path.splitext(p)[1].lower() in ('.jpg','.jpeg','.png')])
    if not paths: return

    for path in paths:
        image = cv2.imread(path)
        if image is None: continue
        plate_res = plate_model(image)[0]
        display_img = image.copy()

        for box in plate_res.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            upscaled = upscale_image(crop, upscaler)
            up = smart_resize(upscaled, 500)
            
            txt_res = text_model(up)[0]
            bxs = txt_res.boxes.xyxy.cpu().numpy()
            conf = txt_res.boxes.conf.cpu().numpy()
            cls = txt_res.boxes.cls.cpu().numpy().astype(int)
            keep = nms(bxs, conf)
            
            kept_boxes = [bxs[i] for i in keep]
            line_groups = get_line_groups(kept_boxes)
            
            all_chars = []
            for group in line_groups:
                line_chars = []
                for box_data in group:
                    box = box_data[0]
                    idx = next(i for i, b in enumerate(kept_boxes) if np.array_equal(b, box))
                    original_idx = keep[idx]
                    line_chars.append((box[0], text_model.names[cls[original_idx]]))
                line_chars.sort(key=lambda x: x[0])
                all_chars.extend([c for _,c in line_chars])
            
            plate_txt = ''.join(all_chars)
            
            for i in keep:
                xA,yA,xB,yB = map(int, bxs[i])
                cv2.rectangle(up, (xA,yA), (xB,yB), (0,255,0),1)
            cv2.putText(up, plate_txt, (10,480), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.rectangle(display_img, (x1,y1), (x2,y2), (0,0,255),2)
            cv2.putText(display_img, plate_txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('Plate Region', up)

        cv2.imshow('Detection', display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()