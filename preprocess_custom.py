import os
import cv2

INPUT_DIR = 'data/kaggle_blurred'      
OUTPUT_DIR = 'data/videos_160'

DIM = 160

def center_crop_and_resize(frame, dim=160):
    """
    Resizes a frame so that its longest side is 'dim', 
    and then center-crops or pads to get exactly dim x dim.
    """
    h, w = frame.shape[:2]
    
    # 1) Scale the frame so the longer side is 'dim'
    ratio = dim / max(h, w)
    new_size = (int(w * ratio), int(h * ratio))  # (width, height)
    resized = cv2.resize(frame, new_size)
    
    # 2) Pad (or center-crop) to get a square dim x dim
    delta_w = dim - new_size[0]
    delta_h = dim - new_size[1]
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    # Using the same border color as the original code (ImageNet means, BGR)
    # But you can use just black or any color.
    b_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT,
                               value=[0.406*255, 0.456*255, 0.485*255])
    return b_img


def preprocess_videos_in_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, dim=DIM):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    for filename in all_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.isfile(output_path):
            print(f"Skipping {filename}, already processed.")
            continue
        
        print(f"Processing {filename} ...")
        cap = cv2.VideoCapture(input_path)
        
        # Get input FPS (fallback to 30 if it can't read)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (dim, dim))
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            # Preprocess each frame
            frame_160 = center_crop_and_resize(frame, dim)
            out.write(frame_160)
        
        cap.release()
        out.release()
        print(f"Saved preprocessed video to {output_path}")


if __name__ == '__main__':
    preprocess_videos_in_folder()
