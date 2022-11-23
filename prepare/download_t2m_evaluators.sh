echo -e "Downloading T2M evaluators"
gdown --fuzzy https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view
rm -rf t2m

unzip t2m.zip
echo -e "Cleaning\n"
rm t2m.zip

echo -e "Downloading done!"