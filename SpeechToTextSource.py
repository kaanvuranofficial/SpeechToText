


# Utils 

class TextProcess:
    """Text processor for converting between text and integer sequences"""
    def __init__(self):
        # Character mapping: a-z (0-25), space (26), apostrophe (27), blank (28)
        self.char_map = {}
        self.int_map = {}
        
        # Create mappings
        chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
                 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        
        self.char_map["'"] = 0
        self.char_map[' '] = 1
        for i, char in enumerate(chars):
            self.char_map[char] = i + 2
        self.char_map['_'] = 28  # blank
        
        # Reverse mapping
        self.int_map = {v: k for k, v in self.char_map.items()}
    
    def text_to_int_sequence(self, text):
        """Convert text string to integer sequence"""
        text = text.lower()
        return [self.char_map.get(c, 1) for c in text if c in self.char_map]
    
    def int_to_text_sequence(self, labels):
        """Convert integer sequence to text string"""
        return ''.join([self.int_map.get(i, '') for i in labels])


# Decoder
import torch

textprocess = TextProcess()

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]


def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
    """Simple greedy CTC decoder"""
    arg_maxes = torch.argmax(output, dim=2).squeeze(1)
    decode = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i - 1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)


class CTCBeamDecoder:
    """CTC Beam Search Decoder with optional KenLM language model"""
    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        print("loading beam search with lm...")
        try:
            import ctcdecode
            self.decoder = ctcdecode.CTCBeamDecoder(
                labels, alpha=0.522729216841, beta=0.96506699808,
                beam_width=beam_size, blank_id=labels.index('_'),
                model_path=kenlm_path)
            print("finished loading beam search")
        except ImportError:
            print("Warning: ctcdecode not installed. Install with: pip install ctcdecode")
            print("Falling back to greedy decoding")
            self.decoder = None
    
    def __call__(self, output):
        if self.decoder is None:
            return DecodeGreedy(output)
        
        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
        return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])
    
    def convert_to_string(self, tokens, vocab, seq_len):
        """Convert tokens to string"""
        return ''.join([vocab[x] for x in tokens[0:seq_len]])


# Model

import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 81,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }
    
    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2),
            ActDropNormCNN1D(n_feats, dropout),
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)
    
    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))
    
    def forward(self, x, hidden):
        x = x.squeeze(1)  # batch, feature, time
        x = self.cnn(x)  # batch, time, feature
        x = self.dense(x)  # batch, time, feature
        x = x.transpose(0, 1)  # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)


# Dataset

import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np


class SpecAugment(nn.Module):
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]
    
    def forward(self, x):
        return self._forward(x)
    
    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x
    
    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x
    
    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            win_length=win_length, hop_length=hop_length)
    
    def forward(self, x):
        x = self.transform(x)  # mel spectrogram
        x = np.log(x + 1e-14)  # logarithmic, add small value to avoid inf
        return x


def get_featurizer(sample_rate, n_feats=81):
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)


class Data(torch.utils.data.Dataset):
    parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15
    }
    
    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                 time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()
        
        print("Loading data json file from", json_path)
        self.data = pd.read_json(json_path, lines=True)
        
        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        try:
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_sequence(self.data['text'].iloc[idx])
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s' % file_path)
            if spectrogram.shape[2] > 1650:
                raise Exception('spectrogram to big. size %s' % spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s' % file_path)
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)
        return spectrogram, label, spec_len, label_len
    
    def describe(self):
        return self.data.describe()


def collate_fn_padd(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)
    
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, input_lengths, label_lengths


# Training

import os
import ast
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class SpeechModule(LightningModule):
    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args
    
    def forward(self, x, hidden):
        return self.model(x, hidden)
    
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=0.50, patience=6)
        return [self.optimizer], [self.scheduler]
    
    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}
    
    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, **d_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        test_dataset = Data(json_path=self.args.valid_file, **d_params, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)


def checkpoint_callback(args):
    return ModelCheckpoint(
        filepath=args.save_model_path,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )


def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)
    model = SpeechRecognition(**h_params)
    
    if args.load_model_from:
        speech_module = SpeechModule.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = SpeechModule(model, args)
    
    logger = TensorBoardLogger(args.logdir, name='speech_recognition')
    
    trainer = Trainer(
        max_epochs=args.epochs, gpus=args.gpus,
        num_nodes=args.nodes, distributed_backend=None,
        logger=logger, gradient_clip_val=1.0,
        val_check_interval=args.valid_every,
        checkpoint_callback=checkpoint_callback(args),
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.fit(speech_module)


# Engine = Main Inference

import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import sys


class Listener:
    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=self.sample_rate,
                                   input=True,
                                   output=True,
                                   frames_per_buffer=self.chunk)
    
    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)
    
    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nSpeech Recognition engine is now listening... \n")


class SpeechRecognitionEngine:
    def __init__(self, model_file, ken_lm_file=None, context_length=10):
        self.listener = Listener(sample_rate=8000)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # run on cpu
        self.featurizer = get_featurizer(8000)
        self.audio_q = list()
        self.hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.beam_results = ""
        self.out_args = None
        self.beam_search = CTCBeamDecoder(beam_size=100, kenlm_path=ken_lm_file)
        self.context_length = context_length * 50  # multiply by 50 because each 50 from output frame is 1 second
        self.start = False
    
    def save(self, waveforms, fname="audio_temp.wav"):
        wf = wave.open(fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(8000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return fname
    
    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)
            log_mel = self.featurizer(waveform).unsqueeze(1)
            out, self.hidden = self.model(log_mel, self.hidden)
            out = torch.nn.functional.softmax(out, dim=2)
            out = out.transpose(0, 1)
            self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
            results = self.beam_search(self.out_args)
            current_context_length = self.out_args.shape[1] / 50  # in seconds
            if self.out_args.shape[1] > self.context_length:
                self.out_args = None
            return results, current_context_length
    
    def inference_loop(self, action):
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)
    
    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                   args=(action,), daemon=True)
        thread.start()


class DemoAction:
    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""
    
    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        transcript = " ".join(self.asr_results.split() + results.split())
        print(transcript)
        if current_context_length > 10:
            self.asr_results = transcript


if __name__ == "__main__":
    parser = ArgumentParser(description="Speech recognition engine demo")
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to optimized model file (.pt)')
    parser.add_argument('--ken_lm_file', type=str, default=None,
                        help='Path to KenLM language model (optional)')
    
    args = parser.parse_args()
    
    asr_engine = SpeechRecognitionEngine(args.model_file, args.ken_lm_file)
    action = DemoAction()
    
    asr_engine.run(action)
    threading.Event().wait()