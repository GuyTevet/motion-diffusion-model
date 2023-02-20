mkdir -p dataset/
cd dataset/

echo "The datasets will be stored in the 'dataset' folder\n"

# HumanAct12 poses
echo "Downloading the HumanAct12 poses dataset"
gdown "https://drive.google.com/uc?id=1130gHSvNyJmii7f6pv5aY5IyQIWc3t7R"
echo "Extracting the HumanAct12 poses dataset"
tar xfzv HumanAct12Poses.tar.gz
echo "Cleaning\n"
rm HumanAct12Poses.tar.gz

# HumanAct12 poses unconstrained
echo "Downloading the HumanAct12 unconstrained poses dataset"
cd HumanAct12Poses
gdown "1KqOBTtLFgkvWSZb8ao-wdBMG7sTP3Q7d"

echo "Downloading done!"
