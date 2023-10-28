import os
import hashlib
import wget

sleepedf_url = "https://www.physionet.org/files/sleep-edfx/1.0.0"
output_dir = os.path.join("./data", "sleepedf")

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

record_file = os.path.join(output_dir, "sleepedf_records.txt")
if os.path.exists(record_file):
    os.remove(record_file)
url = sleepedf_url + "/" + "SHA256SUMS.txt"
wget.download(url, record_file)

recordv1_file = os.path.join(output_dir, "RECORDS-v1")
if os.path.exists(recordv1_file):
    os.remove(recordv1_file)
url = sleepedf_url + "/" + "RECORDS-v1"
wget.download(url, recordv1_file)

with open(recordv1_file) as f:
    list_v1_subjects = [l.strip().split('-')[0][:-2] for l in f.readlines() if 'ST' not in l]

download_files = []
with open(record_file) as f:
    for l in f.readlines():
        l = l.strip()

        tmp = l.split(" ")
        sha256hash = tmp[0]
        fname = tmp[-1]

        if 'sleep-cassette' in fname:
            subject_id = fname.split('/')[1].split('-')[0][:-2]

            # Skip if not in subject version1 (i.e., small dataset)
            if subject_id not in list_v1_subjects:
                continue

            download_url = sleepedf_url + "/" + fname
            save_f = os.path.join(output_dir, fname)
            save_dir = os.path.dirname(save_f)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not os.path.isfile(save_f):
                print(f"\nDownloading {download_url} to {save_f}")
                wget.download(download_url, save_f)
            
            # Check SHA256
            with open(save_f, "rb") as ff:
                b = ff.read()
                readable_hash = hashlib.sha256(b).hexdigest()
                assert sha256hash == readable_hash

                print(f'Downloaded {save_f}')
                download_files.append(save_f)

download_filename = os.path.join("./data", "sleepedf-files.txt")
with open(download_filename, "w") as f:
    f.write("\n".join(download_files))