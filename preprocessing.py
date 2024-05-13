##########################
### raw event (folders of .bin) to event representation (time surface / histogram)

##########################

'''
input preperation:
  path #this is the path to put in raw_input_path augruement
    -class1
      -0001.bin
      -0002.bin
    -class2
      -0001.bin
      -0002.bin
output organization:
  path #this is the path to put in output_path augruement
    -class1
      -0001.pt
      -0002.pt
    -class2
      -0001.pt
      -0002.pt
'''

import click
import tonic
from event_reading import read_event
import tonic.transforms as transforms
import tonic
import numpy as np
import torch
import os
from tqdm import tqdm
import time

@click.command()
@click.option('--raw_input_path', type=str, help='folder contain classes of raw event (.bin) file')
@click.option('--output_path', type=str, help='output directory path')
@click.option('--rep_type', type=click.Choice(['time_surface', 'histogram'], case_sensitive=False))
@click.option('--dt', type=int, default=10000)
@click.option('--num_timebin', type=int, default=10)

def main(raw_input_path, output_path, rep_type, dt, num_timebin):
    if rep_type == 'time_surface':
        event_transform = tonic.transforms.ToTimesurface(
                          sensor_size=(34,34,2),
                          tau=30000,
                          dt=dt,
                          )
    elif rep_type == 'histogram':
        event_transform = tonic.transforms.ToHistogram(
                          sensor_size=(34,34,2),
                          n_time_bins=num_timebin,
                          )
    file_count = 0
    program_starts = time.time()
    now = time.time()
    for root, dirs, files in os.walk(raw_input_path):
        for file in files:
            if file.endswith(".bin"):
                # Prepare input file path
                input_file_path = os.path.join(root, file)
                # Prepare output directory path
                relative_path = os.path.relpath(root, raw_input_path)
                output_dir = os.path.join(output_path, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                # Prepare output file path with .pt extension
                output_file = os.path.splitext(file)[0] + ".pt"
                output_file_path = os.path.join(output_dir, output_file)
                # Load raw file and apply transformation
                raw_file = read_event(input_file_path)
                preprocessed_file = event_transform(raw_file)
                # Save preprocessed file as Torch tensor
                torch.save(preprocessed_file, output_file_path)
                file_count += 1
                now = time.time()
                print(f"Processed file: No.{file_count} in {output_file_path}, time taken = {now - program_starts}")

    return print('Done')

if __name__ == '__main__':
    main()