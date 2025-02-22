import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from eval import ToTensor, Normalize
from model import EventDetector

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

class SampleVideo(Dataset):
    """
    Reads a single .mp4 file frame by frame, resizes/pads to 160x160,
    stores them in a list (as RGB), and returns them as a batch
    with optional transforms (ToTensor, Normalize).
    """
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        # dataset returns a single item (the entire frame sequence)
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # If video can't be read, height/width could be 0
        if frame_height <= 0 or frame_width <= 0:
            print(f"Warning: {self.path} appears unreadable or empty!")
            # return empty arrays so we don't crash
            return {'images': np.empty((0,)), 'labels': np.empty((0,))}

        ratio = self.input_size / max(frame_height, frame_width)
        new_size = (int(frame_width * ratio), int(frame_height * ratio))  # (w, h)

        delta_w = self.input_size - new_size[0]
        delta_h = self.input_size - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(frame_count):
            success, img = cap.read()
            if not success:
                break
            # Resize
            resized = cv2.resize(img, (new_size[0], new_size[1]))  # (w,h) in OpenCV
            # Pad with ImageNet means (BGR)
            b_img = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=[0.406*255, 0.456*255, 0.485*255]
            )
            # Convert to RGB
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)

        cap.release()

        labels = np.zeros(len(images))  # only for transform-compatibility
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def process_video(video_path, model, seq_length=64, device='cpu'):
    """
    Given a single .mp4 video path, runs SwingNet inference
    to detect the 8 events, then returns:
      - events: the frame indices for each event
      - confidence: the probability associated with each event
    """
    # Create a dataset + loader for this one video
    ds = SampleVideo(
        path=video_path,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
        ])
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    # We'll only have 1 batch from ds
    for sample in dl:
        images = sample['images']  # shape: [batch=1, frames, C, H, W]
        if images.shape[1] == 0:
            # Means no frames or broken video
            return None, None

        batch_idx = 0
        all_probs = []
        while batch_idx * seq_length < images.shape[1]:
            start_i = batch_idx * seq_length
            end_i = min((batch_idx + 1) * seq_length, images.shape[1])
            image_batch = images[:, start_i:end_i, :, :, :].to(device)
            logits = model(image_batch)
            probs = F.softmax(logits.data, dim=1).cpu().numpy()
            all_probs.append(probs)
            batch_idx += 1

        # Concatenate along frames dimension
        probs = np.concatenate(all_probs, axis=0)  # shape: [num_frames, 9]
        # Argmax over frames dimension for each of the 9 classes
        # but we only care about the first 8 (the 9th is "no-event")
        events = np.argmax(probs, axis=0)[:-1]  # last is no-event
        confidence = [probs[e, i] for i, e in enumerate(events)]

    return events, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--path',
        help='Path to a single .mp4 file OR a folder containing .mp4 files',
        required=True
    )
    parser.add_argument(
        '-s', '--seq-length',
        type=int,
        help='Number of frames to use per forward pass',
        default=64
    )
    parser.add_argument(
        '-o', '--outdir',
        default='event',
        help='Folder to store output frames for each video'
    )
    args = parser.parse_args()

    # Prepare model
    print('Loading EventDetector model...')
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar')
        model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        print("Loaded swingnet_1800.pth.tar model weights")
    except FileNotFoundError:
        print("❌ Model weights not found at 'models/swingnet_1800.pth.tar'! Exiting.")
        return
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return

    seq_length = args.seq_length
    input_path = args.path
    out_dir = args.outdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Check if `-p` is a folder or a single file
    if os.path.isdir(input_path):
        print(f"Processing all .mp4 files in folder: {input_path}")
        mp4_files = [f for f in os.listdir(input_path) if f.endswith('.mp4')]
        mp4_files.sort()

        for vidfile in mp4_files:
            full_path = os.path.join(input_path, vidfile)
            print(f"\n--- Processing: {vidfile} ---")
            events, confs = process_video(full_path, model, seq_length, device)
            if events is None:
                print(f"⚠️ Skipping {vidfile}, no frames or unreadable.")
                continue

            # Create a sub-folder for each video’s frames
            video_stem = os.path.splitext(vidfile)[0]
            video_outdir = os.path.join(out_dir, video_stem)
            os.makedirs(video_outdir, exist_ok=True)

            print(f"Predicted event frames: {events}")
            print(f"Confidence: {[round(c, 3) for c in confs]}")

            # Save frames
            cap = cv2.VideoCapture(full_path)
            for i, e in enumerate(events):
                cap.set(cv2.CAP_PROP_POS_FRAMES, e)
                ret, img = cap.read()
                if ret:
                    cv2.putText(img,
                                f"{confs[i]:.3f}",
                                (20, 20),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.75,
                                (0, 0, 255),
                                2)
                    outname = os.path.join(video_outdir, f"{event_names[i]}.jpg")
                    cv2.imwrite(outname, img)
            cap.release()

    else:
        # Single file path
        print(f"Processing single .mp4 file: {input_path}")
        # Run inference
        events, confs = process_video(input_path, model, seq_length, device)
        if events is None:
            print(f"⚠️ Skipping {input_path}, no frames or unreadable.")
            return

        # Save frames
        print(f"Predicted event frames: {events}")
        print(f"Confidence: {[round(c, 3) for c in confs]}")

        # Put them in a sub-folder named after the file
        filename = os.path.basename(input_path)
        video_stem = os.path.splitext(filename)[0]
        video_outdir = os.path.join(out_dir, video_stem)
        os.makedirs(video_outdir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            ret, img = cap.read()
            if ret:
                cv2.putText(img,
                            f"{confs[i]:.3f}",
                            (20, 20),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.75,
                            (0, 0, 255),
                            2)
                outname = os.path.join(video_outdir, f"{event_names[i]}.jpg")
                cv2.imwrite(outname, img)
        cap.release()

if __name__ == '__main__':
    main()
