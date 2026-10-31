# BinaryBrains

## Step 01
Add the model directory to the root directory. It should be in the following manner
```
│   .gitignore
│   audio.py
│   encoder.py
│   main.py
│   model.py
│   README.md
│   requirement.txt
│   signal_processing.py
│   __init__.py
│
├───model
       config.json
       optimizer.pt
       preprocessor_config.json
       pytorch_model.bin
       rng_state.pth
       scaler.pt
       scheduler.pt
       trainer_state.json
       training_args (1).bin
       training_args.bin
       vocab.json

```

## Step 02
Set up the virtual environment. 
First make sure python 3.11 is installed and in the path variales
Open the terminal and go to the working directory and run following commands.

```py 3.11 -m venv venv```

After virtual environment is created run
```.\venv\Scripts\activate```

If this gives a scope error try this.
```Set-ExecutionPolicy Unrestricted -Scope Process```
This happens if the terminal is the powershell. You can try to activate activate.ps1 instead.
Refer https://dev.to/shriekdj/python-venv-or-virtualenv-wont-activate-on-windows-3e2

## Step 03
Installing python packages.
After activating the venv run following command to download and install required packages. Make sure you are in the root directory
```pip install -r .\requirements.txt```

## Step 04
Run the main file
```py .\main.py```




# 1. Load the ALSA loopback kernel module (once per boot)
sudo modprobe snd-aloop

# 2. Make it persistent across reboots
echo snd-aloop | sudo tee -a /etc/modules

# 3. Install soundfile if not already present
pip install soundfile


# Terminal 1 — start the I2S mic simulator
python3 i2s_mic_simulator.py --source sine          # 440 Hz L, 880 Hz R
python3 i2s_mic_simulator.py --source noise         # white noise
python3 i2s_mic_simulator.py --source wav --wav my_speech.wav  # real speech
# Terminal 2 — run the main program (reads from the virtual mic)
python3 main_loopback.py
# Optionally, check available devices:
python3 i2s_mic_simulator.py --list-devices