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

# Donwload UESTC poses estimated with VIBE
echo "Downloading the UESTC poses estimated with VIBE"
gdown "https://drive.google.com/uc?id=1LE-EmYNzECU8o7A2DmqDKtqDMucnSJsy"
echo "Extracting the UESTC poses estimated with VIBE"
tar xjvf uestc.tar.bz2
echo "Cleaning\n"
rm uestc.tar.bz2

echo "Downloading done!"
