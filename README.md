# **Wav2Lip** - a modified wav2lip 384 version


Lip-syncing videos using the pre-trained models (Inference)
-------
You can lip-sync any video to any audio:
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result is saved (by default) in `results/result_voice.mp4`. You can specify it as an argument,  similar to several other available options. The audio source can be any file supported by `FFMPEG` containing audio data: `*.wav`, `*.mp3` or even a video file, from which the code will automatically extract the audio.

Train!
----------
There are two major steps: (i) Train the expert lip-sync discriminator, (ii) Train the Wav2Lip model(s).

##### Training the expert discriminator
You can use your own data (with resolution 384x384)

python parallel_syncnet_tanh.py
```
##### Training the Wav2Lip models
You can either train the model without the additional visual quality disriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python parallel_wav2lip_margin.py
```

