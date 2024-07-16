from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording
from tqdm import tqdm
import pandas as pd
import datetime
import librosa
import wave
import os

am = 'AM0016B' # change 

d_path = f'/Volumes/audio_moth/{am}'
meta_path = '/Volumes/audio_moth_data_with_locations_march_2020.xlsx'

dfs = pd.read_excel(meta_path, sheet_name=None)
am_dfs = dfs['Audio moths'].loc[dfs['Audio moths']['Audio file'] == am]
site = am_dfs['Site'].item()
recorder_locations_match = dfs['Recorder locations']['Recorder site'] == site
site_dfs = dfs['Recorder locations'].loc[recorder_locations_match]
lat = site_dfs['GPS long'].values[0]
lon = site_dfs['GPS lat'].values[0]

analyzer = Analyzer()
results = []
ct, recording_start_time = 0, 0

files = os.listdir(os.fsencode(d_path))
files = [os.path.join(d_path, os.fsdecode(f)) for f in files]
files.sort(key=lambda x: os.stat(x).st_birthtime)

for file in tqdm(files):
    stat = os.stat(file)
    bt = stat.st_birthtime
    if file.endswith('.WAV'):
        try:
            with wave.open(file, 'r') as wav_file:
                duration = librosa.get_duration(path=file)
            recording = Recording(
                analyzer,
                file,
                lat=lat,
                lon=lon,
                date=datetime.datetime.fromtimestamp(bt),
                min_conf=0.25)
            recording.analyze()
            if recording.detections:
                for detection in recording.detections:
                    start_time = detection['start_time'] + recording_start_time
                    end_time = detection['end_time'] + recording_start_time
                    confidence = detection['confidence']
                    results.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': confidence})
            recording_start_time += duration

        except:
            break

results_df = pd.DataFrame(results)
results_df.to_csv(f'event_times_{am}.csv', index=False)

