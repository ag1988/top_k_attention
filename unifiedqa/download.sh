# download data from google cloud 

echo "installating gsutil via pip..."
pip install gsutil
echo "installation done..."
mkdir -p data
cd data
mkdir -p preprocessed_datasets
echo "downloading data from google storage gs://unifiedqa/data ..."
gsutil -m cp -r  "gs://unifiedqa/data/*"  preprocessed_datasets
echo "done."