import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
import pickle
import torch.nn.functional as F


# ========== Spectrogram Augmentation Functions ==========
def spec_augment(spec, freq_mask_max=27, time_mask_max=100, n_freq_masks=2, n_time_masks=2):
    """
    Apply SpecAugment: frequency and time masking to spectrogram.

    Args:
        spec: Spectrogram tensor of shape [freq_bins, time_steps]
        freq_mask_max: Maximum width of frequency mask
        time_mask_max: Maximum width of time mask
        n_freq_masks: Number of frequency masks to apply
        n_time_masks: Number of time masks to apply

    Returns:
        Augmented spectrogram
    """
    spec = spec.clone()
    n_freq, n_time = spec.shape

    # Apply frequency masks
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_max)
        f0 = random.randint(0, max(0, n_freq - f))
        spec[f0:f0+f, :] = 0

    # Apply time masks
    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_max, n_time))
        t0 = random.randint(0, max(0, n_time - t))
        spec[:, t0:t0+t] = 0

    return spec


def cutout_augment(spec, n_holes=3, hole_size_freq=20, hole_size_time=20):
    """
    Apply Cutout augmentation: random rectangular region dropout.

    Args:
        spec: Spectrogram tensor of shape [freq_bins, time_steps]
        n_holes: Number of cutout holes to apply
        hole_size_freq: Maximum height of hole in frequency dimension
        hole_size_time: Maximum width of hole in time dimension

    Returns:
        Augmented spectrogram
    """
    spec = spec.clone()
    n_freq, n_time = spec.shape

    for _ in range(n_holes):
        # Random hole size
        h = random.randint(1, hole_size_freq)
        w = random.randint(1, hole_size_time)

        # Random position
        f0 = random.randint(0, max(0, n_freq - h))
        t0 = random.randint(0, max(0, n_time - w))

        # Apply cutout
        spec[f0:f0+h, t0:t0+w] = 0

    return spec


def gaussian_noise_augment(spec, noise_std=0.01):
    """
    Add Gaussian noise to spectrogram.

    Args:
        spec: Spectrogram tensor of shape [freq_bins, time_steps]
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Augmented spectrogram
    """
    noise = torch.randn_like(spec) * noise_std
    return spec + noise


def apply_strong_spec_augmentations(spec, hparams):
    """
    Apply strong augmentations to spectrogram.

    Args:
        spec: Spectrogram tensor of shape [freq_bins, time_steps]
        hparams: Hyperparameters containing augmentation settings

    Returns:
        Augmented spectrogram
    """
    # Check if augmentations should be applied
    if not getattr(hparams, 'use_spec_augment', False):
        return spec

    # Apply with probability
    if random.random() > getattr(hparams, 'spec_augment_prob', 0.5):
        return spec

    spec_aug = spec.clone()

    # Apply SpecAugment
    if getattr(hparams, 'apply_spec_augment', True):
        freq_mask_max = getattr(hparams, 'freq_mask_max', 17)
        time_mask_max = getattr(hparams, 'time_mask_max', 50)
        n_freq_masks = getattr(hparams, 'n_freq_masks', 2)
        n_time_masks = getattr(hparams, 'n_time_masks', 2)
        spec_aug = spec_augment(spec_aug, freq_mask_max, time_mask_max, n_freq_masks, n_time_masks)

    # Apply Cutout
    if getattr(hparams, 'apply_cutout', True):
        n_holes = getattr(hparams, 'cutout_n_holes', 3)
        hole_size_freq = getattr(hparams, 'cutout_hole_size_freq', 20)
        hole_size_time = getattr(hparams, 'cutout_hole_size_time', 20)
        spec_aug = cutout_augment(spec_aug, n_holes, hole_size_freq, hole_size_time)

    # Apply Gaussian noise
    if getattr(hparams, 'apply_gaussian_noise', True):
        noise_std = getattr(hparams, 'gaussian_noise_std', 0.01)
        spec_aug = gaussian_noise_augment(spec_aug, noise_std)

    return spec_aug


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text_path, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text_path)
        self.audiopaths_sid_text_path = audiopaths_sid_text_path
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # Store hparams for augmentation
        self.hparams = hparams

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
  

    # def _filter(self):
    #     """
    #     Filter text & store spec lengths
    #     """
    #     # Store spectrogram lengths for Bucketing
    #     # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    #     # spec_length = wav_length // hop_length

    #     audiopaths_sid_text_new = []
    #     lengths = []
    #     for audiopath, sid, text in self.audiopaths_sid_text:
    #         if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
    #             audiopaths_sid_text_new.append([audiopath, sid, text])
    #             lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
    #     self.audiopaths_sid_text = audiopaths_sid_text_new
    #     self.lengths = lengths
    
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        audiopaths_sid_text_new = []
        lengths = []
        path = self.audiopaths_sid_text_path.replace(".txt", ".balanced.pkl")

        if os.path.exists(path):
            print(f"Loading filtered audiopaths_sid_text from {path}")
            with open(path, "rb") as f:
                combined = pickle.load(f)
            self.audiopaths_sid_text, self.lengths = zip(*combined)
            return None

        min_duration = 0.1  # seconds
        min_frames = int(min_duration * self.sampling_rate / self.hop_length)

        # --- worker ---
        def process_item(item):
            audiopath, sid, age, sex, text = item

            # age filter
            try:
                a = int(age)
            except Exception:
                return None
            if a > 90 or a < 8:
                return None

            # size -> frames
            try:
                frames = os.path.getsize(audiopath) // (2 * self.hop_length)
            except OSError:
                return None

            if frames < min_frames:
                return None

            return ([audiopath, sid, age, sex, text], frames)

        items = list(self.audiopaths_sid_text)

        # Choose a reasonable thread count (I/O bound; 16–64 often good)
        max_workers = 32 #(os.cpu_count() or 8) * 4 #min(32, (os.cpu_count() or 8) * 4)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for out in tqdm(ex.map(process_item, items), total=len(items)):
                if out is not None:
                    results.append(out)

        self.audiopaths_sid_text = [x for x, _ in results]
        self.lengths = [l for _, l in results]
        
        with open(path, "wb") as f:
            pickle.dump(list(zip(self.audiopaths_sid_text, self.lengths)), f)

        print(f"Filtered audiopaths_sid_text saved to {path}")

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, age, sex, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3], audiopath_sid_text[4]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)

        # Apply strong augmentations to create augmented spectrogram
        spec_aug = apply_strong_spec_augmentations(spec, self.hparams)

        sid = self.get_sid(sid)
        age = self.get_age(age)
        sex = self.get_sex(sex)
        return (text, spec, spec_aug, wav, sid, age, sex)

    def get_audio(self, filename):
        if os.path.exists(filename.replace(".wav", "_trimmed.wav")):
            filename = filename.replace(".wav", "_trimmed.wav")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


    def get_sid(self, sid):
        try:
            num = int(sid)
            sid = torch.LongTensor([num])
        except:
            try:
                sid = np.load(sid)
                sid = torch.tensor(sid, dtype=torch.float)
            except Exception as e:
                print(f"Error loading sid {sid}: {e}")
        return sid
    
    def get_age(self, age):
        age = (int(age)//10)-1
        # age = random.randint(age, age + 7)
        age = torch.LongTensor([age])
        return age

    def get_sex(self, sex):
        sex = int(sex)
        sex = torch.LongTensor([sex])
        return sex

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, spec_aug_normalized, wav_normalized, sid, age, sex]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_spec_aug_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        spec_aug_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if batch[0][2].dim() == 1:
            sid = torch.LongTensor(len(batch))
        else:
            sid = torch.FloatTensor(len(batch), 192)

        age = torch.LongTensor(len(batch))
        sex = torch.LongTensor(len(batch))
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        spec_aug_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_aug_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        spec_aug_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            spec_aug = row[2]
            spec_aug_padded[i, :, :spec_aug.size(1)] = spec_aug
            spec_aug_lengths[i] = spec_aug.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[4]

            age[i] = row[5]


            sex[i] = row[6]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_aug_padded, spec_lengths, wav_padded, wav_lengths, sid, age, sex, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_aug_padded, spec_lengths, wav_padded, wav_lengths, sid, age, sex


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size