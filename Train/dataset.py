"""
dataset.py - Korean VSR PyTorch Dataset & DataLoader

Loads preprocessed .npz files and feeds them to PyTorch DataLoader.
Chaplin VideoTransform is applied only during training.
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CHAPLIN_ROOT = "<path/to/chaplin>"
sys.path.insert(0, CHAPLIN_ROOT)


class KoreanVSRDataset(Dataset):
    """Dataset that loads preprocessed .npz files.

    Args:
        data_dir: Path to preprocessed/train/ or preprocessed/test/.
        video_transform: Chaplin VideoTransform instance (train only).
        tokenizer: KoreanTokenizer instance.
        max_frames: Maximum frame count (truncate if exceeded).
    """

    def __init__(self, data_dir, video_transform=None, tokenizer=None, max_frames=None):
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.video_transform = video_transform
        self.tokenizer = tokenizer
        self.max_frames = max_frames

        if not self.file_list:
            print(f"Warning: No .npz files found in {data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Returns:
            video (Tensor): transformed video tensor
            label (Tensor): (L,) long - tokenized integer sequence
            video_length (int): actual frame count
            label_length (int): actual label length
        """
        data = np.load(self.file_list[idx], allow_pickle=True)
        video = data["video"]                # (T, 96, 96) uint8
        text = str(data["text"])

        if self.max_frames and len(video) > self.max_frames:
            video = video[:self.max_frames]

        video = torch.tensor(video, dtype=torch.float32)   # (T, 96, 96)

        if self.video_transform:
            video = self.video_transform(video)
        else:
            video = video.unsqueeze(-1)                    # (T, 96, 96, 1)
            video = video.permute(3, 0, 1, 2)              # (1, T, 96, 96)
            video = video / 255.0
            start = (96 - 88) // 2
            video = video[:, :, start:start+88, start:start+88]   # (1, T, 88, 88)
            video = (video - 0.421) / 0.165

        video_length = video.shape[1]

        label = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        label_length = label.shape[0]

        return video, label, video_length, label_length


def collate_fn(batch):
    """Pad variable-length sequences into a batch.

    Returns:
        videos (Tensor):        (B, 1, T_max, 88, 88)
        labels (Tensor):        (B, L_max)
        video_lengths (Tensor): (B,)
        label_lengths (Tensor): (B,)
    """
    videos, labels, vid_lens, lbl_lens = zip(*batch)

    max_t = max(v.shape[1] for v in videos)
    padded_videos = []
    for v in videos:
        pad_len = max_t - v.shape[1]
        if pad_len > 0:
            pad = torch.zeros(1, pad_len, v.shape[2], v.shape[3], dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        padded_videos.append(v)
    videos = torch.stack(padded_videos)

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    video_lengths = torch.tensor(vid_lens, dtype=torch.long)
    label_lengths = torch.tensor(lbl_lens, dtype=torch.long)

    return videos, labels, video_lengths, label_lengths


def create_dataloaders(data_dir, tokenizer, batch_size=4, num_workers=0, max_frames=None):
    """Create train/test DataLoaders."""
    from pipelines.data.transforms import VideoTransform

    train_dataset = KoreanVSRDataset(
        data_dir=os.path.join(data_dir, "train"),
        video_transform=VideoTransform(speed_rate=1),
        tokenizer=tokenizer,
        max_frames=max_frames,
    )
    test_dataset = KoreanVSRDataset(
        data_dir=os.path.join(data_dir, "test"),
        video_transform=None,
        tokenizer=tokenizer,
        max_frames=max_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
