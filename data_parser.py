from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import os

def read_fsa(filename:str):
    record = SeqIO.read(filename, "abi")
    signal = record.annotations["abif_raw"]
    green_channel=np.array(signal.get('DATA4'))
    orange_channel=np.array(signal.get('DATA205'))

    return green_channel, orange_channel

def plot_signal(signal:np.ndarray,file:str,channel_name:str, save:bool=False, fpath:str=''):
    plt.figure(figsize=(10,8))
    plt.plot(signal, color=channel_name)
    plt.title(f'Electropherogram - {channel_name} Channel - {file}')
    plt.xlabel('Data Points')
    plt.ylabel('Fluorescence Intensity')
    if save:
        plt.savefig(os.path.join(fpath, f'{file}_{channel_name}.png'), dpi=300)
        plt.close()
    else:
        plt.show()

def read_fsa_from_directory(dir):
    # List all files in the directory
    files = os.listdir(dir)
    # Filter for .fsa files
    fsa_files = [f for f in files if f.endswith('.fsa')]
    fsa_files.sort()
    for fsa_file in fsa_files:
        fsa_path = str(os.path.join(dir, fsa_file))
        green_channel, orange_channel = read_fsa(fsa_path)

        save_dir='./Results/signal_plots/'
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name=fsa_file.split('.')[0]
        plot_signal(green_channel,save_name,'green', True, save_dir)
        plot_signal(orange_channel,save_name,'orange', True, save_dir)

def main():
    # Example usage
    #file="../Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/03052025-1A_20250317120227_3-5-25_UCx1_A1_01.fsa"
    read_fsa_from_directory("../Amplicon Length/Test Data/3.17.25 Promega (Redo) UCx #1 03052025/")

if __name__ == "__main__":
    main()