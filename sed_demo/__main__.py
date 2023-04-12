#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module is the main entry point to the app. It contains the specific class
to run the app, and a way of feeding custom parameters through the CLI.

Usage example (ensure that python can ``import sed_demo``):

python -m sed_demo TOP_K=10 TABLE_FONTSIZE=25
"""


from threading import Thread
import os
import argparse
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
from sed_demo import (
    AI4S_BANNER_PATH,
    SURREY_LOGO_PATH,
    CVSSP_LOGO_PATH,
    EPSRC_LOGO_PATH,
    AUDIOSET_LABELS_PATH,
)
from sed_demo.utils import load_csv_labels, LOGGER, colorstr, print_args
from sed_demo.models import Cnn9_GMP_64x64
from sed_demo.audio_loop import AsynchAudioInputStream
from sed_demo.inference import AudioModelInference, PredictionTracker
from sed_demo.gui import DemoFrontend

FILE = Path(__file__).resolve()


class DemoApp(DemoFrontend):
    """
    This class extends the Tk ``DemoFrontend`` with the specific functionality
    to run the sound event detection demo, i.e.:

    1. Instantiate an ``AsynchAudioInputStream`` to write an audio ring buffer
      from the microphone.
    2. Instantiate a ``Cnn9_GMP_64x64`` to detect categories from audio
    3. Instantiate an ``AudioModelInference`` that uses the CNN to periodically
      detect categories from the ring buffer.
    4. Instantiate a ``PredictionTracker`` to filter out undesired categories
      from the CNN output and return the top K, sorted by confidence.
    """

    # custom theme colors
    BG_COLOR = "#fff8fa"
    BUTTON_COLOR = "#ffcc99"
    BAR_COLOR = "#ffcc99"

    def __init__(
        self,
        top_banner_path,
        logo_paths,
        model_path,
        all_labels,
        tracked_labels=None,
        samplerate=32000,
        audio_chunk_length=1024,
        ringbuffer_length=40000,
        model_winsize=1024,
        stft_hopsize=512,
        stft_window="hann",
        n_mels=64,
        mel_fmin=50,
        mel_fmax=14000,
        top_k=6,
        title_fontsize=22,
        table_fontsize=18,
        from_file=False,
        file_name=None,
    ):
        """
        :param top_banner_path: Path to the image showed at the top
        :param logo_paths: list of paths with images showed at the bottom
        :param all_labels: list of categories in same quantity and
          order as used during model training. See files in the ``assets`` dir.
        :param tracked_labels: optionally, a subset of ``all_labels``
          specifying the labels to track (rest will be ignored).
        :param samplerate: Audio samplerate. Ideally it should match the one
          used during model training.
        :param audio_chunk_length: number of samples that the audio recording
          will write at once. Not relevant for the model, but larger chunks
          impose larger delays for the real-time system.
        :param ringbuffer_length: The recorder will continuously update a ring
          buffer. To perform inference, the model will read the whole ring
          buffer, therefore this length determines the duration of the model
          input. E.g. ``length=samplerate`` corresponds to 1 second. Too short
          lengths may miss some contents, too large lengths may take too long
          for real-time computations.
        :param model_winsize: We have waveforms, but the model expects
          a time-frequency representation (log mel spectrogram). This is the
          window size for the STFT and mel operations. Should match training
          settings.
        :param n_mels: Number of mel bins. Should match training settings.
        :param mel_fmin: Lowest mel bin. Should match training settings.
        :param mel_fmax: Highest mel bin. Should match training settings.
        :param top_k: For each prediction, the app will show only the ``top_k``
          categories with highest confidence, in descending order.
        """
        super().__init__(
            top_k,
            top_banner_path,
            logo_paths,
            title_fontsize=title_fontsize,
            table_fontsize=table_fontsize,
        )
        # 1. Input stream from microphone
        self.audiostream = AsynchAudioInputStream(
            samplerate,
            audio_chunk_length,
            ringbuffer_length,
            from_file,
            file_name,
        )
        # 2. DL pretrained model to predict tags from ring buffer
        num_audioset_classes = len(all_labels)
        self.model = Cnn9_GMP_64x64(num_audioset_classes)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint["model"])
        # 3. Inference: periodically read the input stream with the model
        self.inference = AudioModelInference(
            self.model,
            model_winsize,
            stft_hopsize,
            samplerate,
            stft_window,
            n_mels,
            mel_fmin,
            mel_fmax,
        )
        # 4. Tracker: process predictions, return the top K among allowed ones
        self.tracker = PredictionTracker(all_labels, allow_list=tracked_labels)
        #
        self.top_k = top_k
        self.thread = None
        # handle when user closes window
        self.protocol("WM_DELETE_WINDOW", self.exit_demo)

    def inference_loop(self):
        """
        This method is intended to run asynchronously, i.e. in a separate
        thread, when the user presses play. It loops indefinitely, performing
        DL inference and updating the GUI, until the user presses stop. The
        thread stops automatically once this function returns from the loop.
        """
        while self.is_running():
            dl_inference = self.inference(self.audiostream.read())
            top_preds = self.tracker(dl_inference, self.top_k)
            for label, bar, (clsname, pval) in zip(
                self.sound_labels, self.confidence_bars, top_preds
            ):
                label["text"] = clsname
                bar["value"] = pval

    def start(self):
        """
        Starts ring buffer recording and ``inference_loop``, each on its own
        thread.
        """
        self.audiostream.start()
        self.thread = Thread(target=self.inference_loop)
        self.thread.daemon = True
        self.thread.start()  # will end automatically if is_running=False

    def stop(self):
        """
        Stops the ring buffer recording (the inference loop stops as well)
        when user presses stop button.
        """
        # Note that the superclass already handles the update of the
        # ``is_running()`` method, so the thread will stop based on that.
        # Here we only need to stop the audio stream.
        self.audiostream.stop()

    def exit_demo(self):
        """ """
        # if DL inference is running, give order to pause
        if self.is_running():
            print("Waiting for threads to finish...")
            self.toggle_start()
        # thread may take some time to finish, wait to prevent crash
        self.after(0, self.terminate_after_thread)

    def terminate_after_thread(self, wait_loop_ms=50):
        """
        If thread is still alive, wait for ``wait_loop_ms`` and check again.
        Once thread finished, exit app.
        """
        if self.thread is not None and self.thread.is_alive():
            self.after(wait_loop_ms, self.terminate_after_thread)
        else:
            print("Exiting...")
            self.audiostream.terminate()
            self.destroy()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=32000, help="sampling rate of source")
    parser.add_argument(
        "--chunk", type=int, default=1024, help="sampling rate of source"
    )
    parser.add_argument(
        "--windows", type=int, default=1024, help="sampling rate of source"
    )
    parser.add_argument("--topk", type=int, default=6, help="sampling rate of source")
    parser.add_argument(
        "--from-file", action="store_true", help="compile model (GPU ONLY)"
    )
    parser.add_argument(
        "--name", default="sample.wav", help="save results to project/name"
    )
    parser.add_argument(
        "--pth",
        default="Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth",
        help="save results to project/name",
    )

    opt = parser.parse_args()
    print_args(FILE.stem, opt)

    return opt


@dataclass
class ConfDef:
    """
    Check ``DemoApp`` docstring for details on the parameters. Defaults should
    work reasonably well out of the box.
    """

    def __init__(self, opt):
        self.ROOT = str(FILE.parents[1])
        self.SAMPLERATE = opt.sr
        self.AUDIO_CHUNK_LENGTH = opt.chunk
        self.MODEL_WINSIZE = opt.windows
        self.RINGBUFFER_LENGTH: int = int(self.SAMPLERATE * 2)
        self.TOP_K = opt.topk
        self.from_file = opt.from_file
        self.file_path = os.path.join(self.ROOT, "sed_demo", opt.name)  # absolute path
        self.pth = opt.pth
        self.MODEL_PATH: str = os.path.join(
            self.ROOT, "sed_demo", "models", self.pth
        )  # absolute path

    ALL_LABELS_PATH: str = AUDIOSET_LABELS_PATH
    SUBSET_LABELS_PATH: Optional[str] = None
    STFT_HOPSIZE: int = 512
    STFT_WINDOW: str = "hann"
    N_MELS: int = 64
    MEL_FMIN: int = 50
    MEL_FMAX: int = 14000

    # frontend
    TITLE_FONTSIZE: int = 28
    TABLE_FONTSIZE: int = 22


if __name__ == "__main__":
    print("\n\nCONFIGURATION:")
    # print(ROOT)
    opt = parse_opt()
    CONF = ConfDef(opt)

    _, _, all_labels = load_csv_labels(CONF.ALL_LABELS_PATH)
    if CONF.SUBSET_LABELS_PATH is None:
        subset_labels = None
    else:
        _, _, subset_labels = load_csv_labels(CONF.SUBSET_LABELS_PATH)
    logo_paths = [SURREY_LOGO_PATH, CVSSP_LOGO_PATH, EPSRC_LOGO_PATH]

    demo = DemoApp(
        AI4S_BANNER_PATH,
        logo_paths,
        CONF.MODEL_PATH,
        all_labels,
        subset_labels,
        CONF.SAMPLERATE,
        CONF.AUDIO_CHUNK_LENGTH,
        CONF.RINGBUFFER_LENGTH,
        CONF.MODEL_WINSIZE,
        CONF.STFT_HOPSIZE,
        CONF.STFT_WINDOW,
        CONF.N_MELS,
        CONF.MEL_FMIN,
        CONF.MEL_FMAX,
        CONF.TOP_K,
        CONF.TITLE_FONTSIZE,
        CONF.TABLE_FONTSIZE,
        CONF.from_file,
        CONF.file_path,
        # str(ROOT).replace("\\", "/") + "/sed_demo/" + CONF.file_name
    )

    demo.mainloop()
