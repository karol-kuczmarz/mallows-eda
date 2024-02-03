import requests
from mallows import config
import tarfile
import os
import shutil
import gzip

url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
response = requests.get(url, stream=True)

with open(config.DATA_DIR / "ALL_tsp.tar.gz", "wb") as f:
    f.write(response.content)

shutil.rmtree(config.DATA_DIR / "tsp", ignore_errors=True)
os.mkdir(config.DATA_DIR / "tsp")

with tarfile.open(config.DATA_DIR / "ALL_tsp.tar.gz", "r:gz") as tar:
    tar.extractall(config.DATA_DIR / "tsp")


os.remove(config.DATA_DIR / "ALL_tsp.tar.gz")
files = os.listdir(config.DATA_DIR / "tsp")
for file in files:
    if file.endswith(".gz"):
        with gzip.open(config.DATA_DIR / "tsp" / file, "rb") as tar:
            with open(config.DATA_DIR / "tsp" / file[:-3], "wb") as f:
                f.write(tar.read())

for file in files:
    if file.endswith(".gz"):
        os.remove(config.DATA_DIR / "tsp" / file)