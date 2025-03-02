# What is Adaptive-ComCAT
![image](https://github.com/user-attachments/assets/f76eef9c-f4a1-4920-b428-704dcf0bd439)

The input image is segmented into patches of the sequence and sent to the transformer encoder by position coding. Each block has 1 attention layer, corresponding to two weight matrices WQK and WVO, and 2 full connection layers, corresponding to W1 and W2. After SVD low-rank decomposition, the original weight matrix is replaced. After multi layer perceptron (MLP), the function of image classification can be realized. By monitoring the difference of Rank matrix, the frequency coefficient in the epoch t-1 iteration, i.e. F(t-1), is transferred to the epoch t, i.e. F(t), stage through the dynamic programming method, so as to reduce the use frequency of NAS and accelerate the effect of model training.
# Setup
