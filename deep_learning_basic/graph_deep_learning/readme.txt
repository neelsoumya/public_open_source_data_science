graph deep learning using Node2Vec and dgl
 


https://github.com/aditya-grover/node2vec


python3 -m venv ~/.venvs/venv_node2vec  # create the venv
source ~/.venvs/venv_node2vec/bin/activate


pip3 install scipy pandas networkx gensim numpy

pip3 install dgl rdflib torch tensorflow keras matplotlib

https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html#sphx-glr-download-tutorials-models-1-gnn-1-gcn-py

# download this code

python3 1_gcn.py

python3 zachary_karate_club.py # zachary karate club


deactivate


# simple example
https://docs.dgl.ai/en/latest/tutorials/basics/1_first.html



python3 -m venv ~/.venvs/venv_node2vec2  # create the venv
source ~/.venvs/venv_node2vec/bin/activate

pip3 install graph_nets tensorflow_gpu tensorflow_probability matplotlib scipy


# Graph autoencoder from Kipf and Welling
# Download from
# https://github.com/tkipf/gae
# unzip and go to directory
python3 setup.py install
cd gae
python train.py
python train.py --dataset citeseer 
