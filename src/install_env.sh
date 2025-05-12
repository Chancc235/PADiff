# 0 ==========> linux dependencies <==========
sudo apt install -y --upgrade xvfb 

# 1 ==========> base environments <==========
conda create -p ~/anaconda3/envs/pymarl python=3.7 -y
~/anaconda3/envs/pymarl/bin/python -m pip install sacred numpy scipy matplotlib seaborn pyyaml==5.4.1 pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx nvidia-ml-py3 cloudpickle tqdm opencv-python gym==0.21.0 scikit-learn
conda install -n pymarl pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
~/anaconda3/envs/pymarl/bin/python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# 3 ==========> MARL environments <==========
~/anaconda3/envs/pymarl/bin/python -m pip install git+https://github.com/oxwhirl/smac.git
~/anaconda3/envs/pymarl/bin/python -m pip install -e src/envs/overcooked/overcooked_ai
~/anaconda3/envs/pymarl/bin/python -m pip install -e src/envs/lbf/lb-foraging

# 4 ==========> SC2 game <==========
SC2_FILE="SC2.4.6.2.69232.zip"
SC2PATH="$HOME/StarCraftII"
if [ ! -d "$SC2PATH" ]; then
  mkdir "$SC2PATH"
fi
wget -c http://blzdistsc2-a.akamaihd.net/Linux/$SC2_FILE
unzip -P iagreetotheeula $SC2_FILE
shopt -s dotglob
mv StarCraftII/* "$HOME/StarCraftII/"
rm -rf $SC2_FILE
rm -rf StarCraftII
echo "export SC2PATH=$SC2PATH" >>~/.zshrc # set to your specific shell
echo "SC2PATH is set to: $SC2PATH"

# 5 ==========> SMAC Maps <==========
MAP_DIR="$SC2PATH/Maps/SMAC_Maps/"
if [ ! -d $MAP_DIR ]; then
    mkdir -p $MAP_DIR
fi
cp -r src/envs/starcraft2/maps/SMAC_Maps/* $MAP_DIR
echo "MAP_DIR is set to: $MAP_DIR"
