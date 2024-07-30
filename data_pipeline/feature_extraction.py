import numpy as np
import os
import librosa
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from my_utils import MyUtils
import matplotlib.pyplot as plt
from PIL import Image

# https://librosa.org/doc/latest/feature.html
class FeatureExtraction:
    '''
    Class to extract relevant audio features from each audio file.
    '''
    def __init__(this, sounds_location):
        this.sounds_location = sounds_location
        this.animal_and_noises = {}
        this.animal_and_locations = {}
        this.animal_noise_data = {}
        this.animal_stft_data = {}
        this.animal_stats = {}
        this.df = None

    def convert_to_mono(this, animal, noises):
        '''
           Converts each audio file to mono. This was done out of ignorance.
           
           
            Args:
                animal (str) : The original values from the df as a last.
                                  in this case the species.
                noises (list) : Corresponding audio files for each animal.
            Returns:
                None
            Raises:
                Not yet implemented
        '''
        locations = []
        print(this.sounds_location)
        # used when extracting features from the sound being predicted
        if(this.sounds_location == "predict"):
            try:
                    location = "predict/predict.wav"
                    audio, sample_rate = librosa.load(location, mono=True)

                    # Resave the audio file with a single channel
                   # sf.write(location, audio, sample_rate)

                    locations.append(location)
                # error handling as some sounds can't be loaded for some reason
            except Exception as e:
                raise(e)
        elif(this.sounds_location == "Animal-Sound-Dataset-master"):
            for noise in noises:
                location = f"{this.sounds_location}/{animal}/{noise}"
                try:
                    # AudioSegment is rubbish and has been removed when possible
                    sound = AudioSegment.from_wav(location)
                    sound = sound.set_channels(1)
                    sound.export(location, format="wav")
                    locations.append(location)
                except Exception as e:
                    print(e)
                    continue
        # mapping each audio file to its animal
        this.animal_and_locations[animal] = locations
        print(locations)


    def remove_silence(this, input_file, output_file):
        '''
            Removing silence from audio files as a way to improve data quality.

            Args:
                input_file (str) : The original audio file
                output_file (list) : The new audio file without the silence
            Returns:
                None
            Raises:
                Not yet implemented

        '''
        #print("remove_silence reached")
        audio = AudioSegment.from_wav(input_file)
        # min_silence_len is in ms, silence_thresh in dBFS
        chunks = split_on_silence(audio, min_silence_len=200, silence_thresh=-37.5)
        # creating the new file
        output = AudioSegment.empty()
        for chunk in chunks:
            output += chunk
        output.export(output_file, format="wav")


    def get_noise_data(this, animal):
        '''
            Extracting time series audio data from audio file using librosa.


            Args:
                animal (str) : type of animal. used as a key to get the
                relevant file locations and used as a key in a new 
                dictionary mapping the animal to the corresponding 
                audio data
            Returns:
                None
            Raises:
                Not yet implemented

        '''
        print("get_noise_data reached")
        datas = []
        for location in this.animal_and_locations[animal]:
            data, _ = librosa.load(location,sr=44100,mono=True)
            datas.append(data)
        this.animal_noise_data[animal] = datas
        print(datas)

    def create_spectrograms(this):
        for animal in this.animal_noise_data:
            print(2)
            count = 0
            print(this.animal_noise_data)
            for y in this.animal_noise_data[animal]:
                print(y)
                # removing animals with two parts due to the amount of data attached
                ignore = ['Kedi-Part2','Kopek-Part2','Kus-Part2']
                if(animal in ignore or count == 66):
                    break
                stft_noise = np.abs(librosa.stft(y))
                print('stft',stft_noise)
                img = librosa.display.specshow(librosa.amplitude_to_db(stft_noise,ref=np.max),sr=44100)
                #https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                if(this.sounds_location == "predict"):
                    print("hi")
                    plt.savefig("./predict_image/predict/predict.png", bbox_inches = 'tight',
                pad_inches = 0)
                else:
                    # new empty folder just in case verification is need
                    plt.savefig("./mel_images/" + animal +"/" + animal + "_" + str(count) + ".png", bbox_inches = 'tight',
                pad_inches = 0)
                count += 1
                #plt.show()
            #this.animal_stft_data[x]





    def calculate_statistical_features(this):
       '''
            Calculating various audio features from the audio time series
            Basic statitiscal calculations (mean,median, variance, std in most cases ) 
            are then performed to extract some meaning.
            Takes a long time to run. 
            Args:
                None
            Returns:
                None
            Raises:
                Not yet implemented

        '''
       print("calculate_statistical_features reached")
       # higher sample rate in order to collect more data
       SAMPLE_RATE = 44_100
       for x in this.animal_noise_data:
            # following empty arrays are waiting to be 
            # populated by the relevant stat
            # for each audio file for the current
            # animal in the iteration
            rms_s_mean = []
            rms_s_median = []
            rms_s_var = []
            rms_s_std = []

            chroma_stft_s_mean = []
            chroma_stft_s_median = []
            chroma_stft_s_var = []
            chroma_stft_s_std = []

            bandwidth_s_mean = []
            bandwidth_s_median = []
            bandwidth_s_var = []
            bandwidth_s_std = []

            flatness_s_mean = []
            flatness_s_median = []
            flatness_s_var = []
            flatness_s_std = []

            rolloff_max_s_mean = []
            rolloff_max_s_median = []
            rolloff_max_s_var = []
            rolloff_max_s_std = []

            rolloff_min_s_mean = []
            rolloff_min_s_median = []
            rolloff_min_s_var = []
            rolloff_min_s_std = []

            mel_s_mean = []
            mel_s_median = []
            mel_s_var = []
            mel_s_std = []

            contrast_s_mean = []
            contrast_s_median = []
            contrast_s_var = []
            contrast_s_std = []

            pitch_freq_mean = []
            pitch_freq_var = []
            pitch_freq_std = []

            pitch_mag_mean = []
            # pitch_mag_median = []
            pitch_mag_var = []
            pitch_mag_std = []

            zcr_mean = []
            zcr_median = []
            zcr_var = []
            zcr_std = []

            # hnr_mean = []
            # hnr_median = []
            # hnr_var = []
            hnr_std = []

            spectral_c_mean = []
            spectral_c_median = []
            spectral_c_var = []
            spectral_c_std = []

            for noise in this.animal_noise_data[x]:
                # disregarding any null data
                if len(noise) != 0:
                   # simplified descriptions of each feature
                   # both var and std may not be needed. one or the 
                   # other would probably suffice     

                    # measure of power of the audio                       
                    rms_list = librosa.feature.rms(y=noise) 
                    rms_s_mean.append(np.mean(rms_list))
                    rms_s_median.append(np.median(rms_list))
                    rms_s_var.append(np.var(rms_list))
                    rms_s_std.append(np.std(rms_list))
                    
                    # ends up with the concentrations of similar frequencies over time 
                    chroma_list = librosa.feature.chroma_stft(y=noise, sr=SAMPLE_RATE) 
                    chroma_stft_s_mean.append(np.mean(chroma_list))
                    chroma_stft_s_median.append(np.median(chroma_list))
                    chroma_stft_s_var.append(np.var(chroma_list))
                    chroma_stft_s_std.append(np.std(chroma_list))
                    
                    # difference between highest and lowest frequencies
                    bandwidth_list = librosa.feature.spectral_bandwidth(y=noise, sr=SAMPLE_RATE)
                    bandwidth_s_mean.append(np.mean(bandwidth_list))
                    bandwidth_s_median.append(np.median(bandwidth_list))
                    bandwidth_s_var.append(np.var(bandwidth_list))
                    bandwidth_s_std.append(np.std(bandwidth_list))
                    
                    # variation of the intensity against frequency
                    flatness_list = librosa.feature.spectral_flatness(y=noise)
                    flatness_s_mean.append(np.mean(flatness_list))
                    flatness_s_median.append(np.median(flatness_list))
                    flatness_s_var.append(np.var(flatness_list))
                    flatness_s_std.append(np.std(flatness_list))
                    
                    # max frequency where x% of the energy is below 
                    rolloff_max_list = librosa.feature.spectral_rolloff(y=noise, sr=SAMPLE_RATE, roll_percent=0.99)
                    rolloff_max_s_mean.append(np.mean(rolloff_max_list))
                    rolloff_max_s_median.append(np.median(rolloff_max_list))
                    rolloff_max_s_var.append(np.var(rolloff_max_list))
                    rolloff_max_s_std.append(np.std(rolloff_max_list))

                    rolloff_min_list = librosa.feature.spectral_rolloff(y=noise, sr=SAMPLE_RATE, roll_percent=0.01)
                    rolloff_min_s_mean.append(np.mean(rolloff_min_list))
                    rolloff_min_s_median.append(np.median(rolloff_min_list))
                    rolloff_min_s_var.append(np.var(rolloff_min_list))
                    rolloff_min_s_std.append(np.std(rolloff_min_list))

                    
                    mel_s_list = librosa.feature.mfcc(y=noise, sr=SAMPLE_RATE)
                    mel_s_mean.append(np.mean(mel_s_list))
                    mel_s_median.append(np.median(mel_s_list))
                    mel_s_var.append(np.var(mel_s_list))
                    mel_s_std.append(np.std(mel_s_list))

                    # gathers the peaks and the magnitudes of those peaks
                    pitch_freq_list, pitch_mag_list = librosa.piptrack(y=noise, sr=SAMPLE_RATE)
                    pitch_freq_mean.append(np.mean(pitch_freq_list))
                    pitch_freq_var.append(np.var(pitch_freq_list))
                    pitch_freq_std.append(np.std(pitch_freq_list))

                    pitch_mag_mean.append(np.mean(pitch_mag_list))
                    pitch_mag_var.append(np.var(pitch_mag_list))
                    pitch_mag_std.append(np.std(pitch_mag_list))

                    # rate at which the signal changes direction    
                    zcr_list = librosa.feature.zero_crossing_rate(y=noise)
                    zcr_mean.append(np.mean(zcr_list))
                    zcr_median.append(np.median(zcr_list))
                    zcr_var.append(np.var(zcr_list))
                    zcr_std.append(np.std(zcr_list))

                    # weird noise you get when blowing into a bottle
                    hnr_list = librosa.effects.harmonic(y=noise)
                    # hnr_mean.append(np.mean(hnr_list))
                    # hnr_median.append(np.median(hnr_list))
                    # hnr_var.append(np.var(hnr_list))
                    hnr_std.append(np.std(hnr_list))

                    # 
                    S = np.abs(librosa.stft(noise))
                    # decibel variation across the same frequency range over time
                    contrast_s_mean.append(np.mean(librosa.feature.spectral_contrast(S=S, sr=SAMPLE_RATE).mean()))
                    contrast_s_median.append(np.median(librosa.feature.spectral_contrast(S=S, sr=SAMPLE_RATE)))
                    contrast_s_var.append(np.var(librosa.feature.spectral_contrast(S=S, sr=SAMPLE_RATE)))
                    contrast_s_std.append(np.std(librosa.feature.spectral_contrast(S=S, sr=SAMPLE_RATE)))

                    # spectrums centre of mass    
                    spectral_c_list = librosa.feature.spectral_centroid(y=noise, sr=SAMPLE_RATE)
                    spectral_c_mean.append(np.mean(spectral_c_list))
                    spectral_c_median.append(np.median(spectral_c_list))
                    spectral_c_var.append(np.var(spectral_c_list))
                    spectral_c_std.append(np.std(spectral_c_list))
                    
                # probably not very efficient but it works
                #    current animal is then assigned the list of the calculated features
                 
                this.animal_stats[x] = {'rms_s_mean': rms_s_mean, 
                                'rms_s_median' : rms_s_median,
                                'rms_s_var' : rms_s_var,
                                'rms_s_std':rms_s_std,
                                'chroma_stft_s_mean': chroma_stft_s_mean,
                                'chroma_stft_s_median':chroma_stft_s_median,
                                'chroma_stft_s_var' : chroma_stft_s_var,
                                'chroma_stft_s_std' : chroma_stft_s_std,
                                "bandwidth_s_mean":bandwidth_s_mean,
                                'bandwidth_s_median': bandwidth_s_median,
                                'bandwidth_s_var': bandwidth_s_var, 
                                'bandwidth_s_std': bandwidth_s_std, 
                                'flatness_s_mean':flatness_s_mean,
                                'flatness_s_median' : flatness_s_median,
                                'flatness_s_var': flatness_s_var,
                                'flatness_s_std': flatness_s_std,
                                'rolloff_max_s_mean': rolloff_max_s_mean,
                                'rolloff_max_s_median':rolloff_max_s_median,
                                'rolloff_max_s_var' : rolloff_max_s_var,
                                'rolloff_max_s_std' : rolloff_max_s_std,
                                'rolloff_min_s_mean' : rolloff_min_s_mean,
                                'rolloff_min_s_median' : rolloff_min_s_median,
                                'rolloff_min_s_var': rolloff_min_s_var,
                                'rolloff_min_s_std': rolloff_min_s_std,
                                "mel_s_mean":mel_s_mean,
                                'mel_s_median': mel_s_median,
                                'mel_s_var' : mel_s_var,
                                'mel_s_std' : mel_s_std,
                                'contrast_s_mean' : contrast_s_mean,
                                'contrast_s_median':contrast_s_median,
                                'contrast_s_var' : contrast_s_var,
                                'contrast_s_std' : contrast_s_std,
                                'pitch_freq_mean' : pitch_freq_mean,
                                'pitch_freq_var' : pitch_freq_var,
                                'pitch_freq_std' : pitch_freq_std,
                                'pitch_mag_mean' : pitch_mag_mean,
                                'pitch_mag_var' : pitch_mag_var,
                                'pitch_mag_std' : pitch_mag_std,
                                'zcr_mean' : zcr_mean,
                                'zcr_median' : zcr_median,
                                'zcr_var' : zcr_var,
                                'zcr_std' : zcr_std,
                                # removed as they only give zero vals
                                # 'hnr_mean' : hnr_mean,
                                # 'hnr_median' : hnr_median,
                                # 'hnr_var' : hnr_var,
                                'hnr_std' : hnr_std,
                                'spectral_c_mean' : spectral_c_mean,
                                'spectral_c_median' : spectral_c_median,
                                'spectral_c_var' : spectral_c_var,
                                'spectral_c_std' : spectral_c_std
                                }

    def process_sounds(this):
        '''
            Runner
        '''
        animals = os.listdir(this.sounds_location)
        #animals.remove("README.md")

        for animal in animals:
            location = ""
            #location = f"{this.sounds_location}/{animal}"
            # single version
            if(this.sounds_location == "predict"):
                print("noise")
                location = f"{this.sounds_location}"
                print(location)
            elif(this.sounds_location ==  "Animal-Sound-Dataset-master"):
                location = f"{this.sounds_location}/{animal}"
            this.animal_and_noises[animal] = os.listdir(location)
            this.convert_to_mono(animal, this.animal_and_noises[animal])

        for animal in this.animal_and_locations.keys():
            for sound in this.animal_and_locations[animal]:
                if(this.sounds_location != "predict"):
                    this.remove_silence(sound, sound)

        for animal in this.animal_and_locations.keys():
            this.get_noise_data(animal)

        this.calculate_statistical_features()
    
    def spec_run(this):
        '''
            Runner
        '''
        animals = os.listdir(this.sounds_location)
        #animals.remove("README.md")

        for animal in animals:
            location = ""
            #location = f"{this.sounds_location}/{animal}"
            # single version
            if(this.sounds_location == "predict"):
                location = f"{this.sounds_location}"
            elif(this.sounds_location ==  "Animal-Sound-Dataset-master"):
                location = f"{this.sounds_location}/{animal}"
            this.animal_and_noises[animal] = os.listdir(location)
            this.convert_to_mono(animal, this.animal_and_noises[animal])

        for animal in this.animal_and_locations.keys():
            for sound in this.animal_and_locations[animal]:
                if(this.sounds_location != "predict"):
                    this.remove_silence(sound, sound)

        for animal in this.animal_and_locations.keys():
            this.get_noise_data(animal)
        print(this.animal_noise_data)
        this.create_spectrograms()
        this.calculate_statistical_features()
        
if __name__ == "__main__":
    # n.b. this takes a long time on my machine. all data will be generated in the submission.
    choice = input('Enter "acoustical" to generate acoustical features, or "spectrogram" to generate images, alternatively type "both" to run both: ')
    choice = choice.lower()
    choice = choice.strip()
    sounds_location = "Animal-Sound-Dataset-master"
    if(choice == "acoustical"):
        extractor = FeatureExtraction(sounds_location)
        extractor.process_sounds()
        df = MyUtils.create_df(extractor.animal_stats)
        MyUtils.data_frame_to_csv(df,"raw_sound_stats_verify.csv")
    elif(choice == "spectrogram"):
        extractor = FeatureExtraction(sounds_location)
        extractor.spec_run()
    elif(choice == "both"):
        # extractor = FeatureExtraction(sounds_location)
        # extractor.spec_run()
        extractor = FeatureExtraction(sounds_location)
        extractor.process_sounds()
        df = MyUtils.create_df(extractor.animal_stats)
        MyUtils.data_frame_to_csv(df,"raw_sound_stats_verify.csv")
        extractor.spec_run()

