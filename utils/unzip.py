import zipfile

with zipfile.ZipFile("/opt/ml/outputs/13ddzwb9.zip") as z:
    z.extractall()