FROM tensorflow/tensorflow:2.3.1
MAINTAINER younes mueller "ymueller@aices.rwth-aachen.de"
 
RUN pip install tensorflow==2.3.1 tensorflow-probability==0.11.1 casadi numpy matplotlib seaborn tqdm humanfriendly flake8 openpyxl