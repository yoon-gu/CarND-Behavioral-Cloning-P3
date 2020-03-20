import csv
import os
import shutil

src_list = ['normal_lap',
            'backward_lap',
            'second_track',
            ]

dst = 'data'
dst_path = os.path.join('./', dst, 'IMG')
os.mkdir(dst_path)

for src in src_list:
    src_path = os.path.join('./', src, 'IMG')
    print(src_path)
    src_files = os.listdir(src_path)
    print(len(src_files))
    for file_name in src_files:
        full_file_name = os.path.join(src_path, file_name)
        if (os.path.isfile(full_file_name)):
            # pirnt(full_file_name)
            shutil.copy(full_file_name, dst_path)



csv_filename = 'driving_log.csv'
dst_csv = os.path.join('./', dst, csv_filename)
with open(dst_csv, 'w') as outfile:
    for src in src_list:
        fname = os.path.join('./', src, csv_filename)
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)