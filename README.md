# Neural Network in NumPy

This is an in depth explanation and implementation of a simple feed-forward neural net using NumPy.

## Feed-Forward Neural Network

Note: This notebook already assumes a basic knowledge of neural nets. Things like layers and layer sizes, activation functions, batching, and so on.

By the end of the notebook we are going to create a simple feed-forward neural net that learns to recognize handwritten digits using the [MNIST-dataset](http://yann.lecun.com/exdb/mnist/).

We'll first start by training a simple neural network to learn to classify XOR ($\oplus$):

<table>
    <thead><tr><td>a</td><td>b</td><td>$a \oplus b$</td></tr></thead>
    <tbody>
        <tr><td>0</td><td>0</td><td>0</td></tr>
        <tr><td>0</td><td>1</td><td>1</td></tr>
        <tr><td>1</td><td>0</td><td>1</td></tr>
        <tr><td>1</td><td>1</td><td>0</td></tr>
    </tbody>
</table>

---

We'll start by defining the structure of our network:
<div style="text-align: center;">
    <img src="static/2-3-1-nn.png" width="60%">
</div>

- The first layer (aka the input layer) has two inputs corresponding to $a$ and $b$.
- The middle / hidden layer is composed of three neurons.
- The final layer (aka the output layer) has a single output. 

The output of the neural network is a prediction that is the probability of the instance to belong to the positive class.
So anything above 0.5 we could classify as 1 and everthing below 0.5 we could classify as 0.

## Feed-Forward

It's called a **Feed-Forward Neural Net** because we **feed the input forward** through the network starting at the input layer until the output.

Here's how we implement the feed forward algorithm.

$$
\mathbf{Z}_1 = \mathbf{X}_1 \cdot \mathbf{W}_1 \\
\mathbf{X}_2 = \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{Z}_2 = \mathbf{X}_2 \cdot \mathbf{W}_2 \\
\mathbf{\hat{Y}} = \sigma(\mathbf{Z_2})
$$

Note: $\large \cdot$ represents matrix multiplication.

To start we'll get some notation out of the way:
1. $\mathbf{X}_1$ is the input. 
    - It can either be a single instance i.e. \[0, 0\] (1 x 2) or a batch of instances \[[0,0],[0,1],[1,0]] (3 x 2)
1. $\mathbf{W}_1$ is the first weight matrix with a shape of (2 x 3)
1. $\mathbf{W}_2$ is the second weight matrix with a shape of (3 x 2)

- We forward our input through the first layer and get out $\mathbf{Z}_1$. 
- We then apply a ReLU activation function on $\mathbf{Z}_1$ and get $\mathbf{X}_2$. 
    - We then forward $\mathbf{X}_2$ through the second layer and get $\mathbf{Z}_2$.
- Finally we apply sigmoid ($\sigma$) on $\mathbf{Z}_2$ to get a probability, $\mathbf{\hat{Y}}$.

## Backpropagation

In backpropagation, we learn what the right set of weights are in order to give the desired output.

In this example assume our network is a general 3 layer network like below (instead of 2). This will help illustrate the pattern that arises for backpropagation. 

<div style="text-align: center;">
    <img src="static/3-layer-nn.png" width="60%" />
</div>

We assume we have a cost function (denoted $J$, in this case cross-entropy loss). We find the partial derivatives of the cost function with respect to each weight matrix: 

$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_3} =  \frac{\partial \mathbf{J}}{\partial \mathbf{\hat{Y}}} \cdot \frac{\partial \mathbf{\hat{Y}}}{\partial \mathbf{Z}_3} \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{W}_3}
$$

$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_2} = \frac{\partial \mathbf{J}}{\partial \mathbf{\hat{Y}}} \cdot \frac{\partial \mathbf{\hat{Y}}}{\partial \mathbf{Z}_3} \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{X}_3} \cdot \frac{\partial \mathbf{X}_3}{\partial \mathbf{Z}_2} \cdot \frac{\partial \mathbf{Z}_2}{\partial \mathbf{W}_2}
$$

$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_1} = \frac{\partial \mathbf{J}}{\partial \mathbf{\hat{Y}}} \cdot \frac{\partial \mathbf{\hat{Y}}}{\partial \mathbf{Z}_3} \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{X}_3} \cdot \frac{\partial \mathbf{X}_3}{\partial \mathbf{Z}_2} \cdot \frac{\partial \mathbf{Z}_2}{\partial \mathbf{X}_2} \cdot \frac{\partial \mathbf{X}_2}{\partial \mathbf{Z}_1} \cdot \frac{\partial \mathbf{Z}_1}{\partial \mathbf{W}_1}
$$

**Let's break down the formulas:**

$\large \frac{\partial \mathbf{J}}{\partial \mathbf{W}_3}$:

$\text{We'll set }\delta_3 = \frac{\partial \mathbf{J}}{\partial \mathbf{\hat{Y}}} \cdot \frac{\partial \mathbf{\hat{Y}}}{\partial \mathbf{Z}_3} = \mathbf{\hat{Y}} - \mathbf{Y}$

Since $\mathbf{Z}_3 = \mathbf{X}_3 \cdot \mathbf{W}_3 \to \frac{\partial \mathbf{Z}_3}{{\partial \mathbf{W}_3}} = \mathbf{X}_3$ So in total:

$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_3} = \frac{\partial \mathbf{J}}{\partial \mathbf{\hat{Y}}} \cdot \frac{\partial \mathbf{\hat{Y}}}{\partial \mathbf{Z}_3} \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{W}_3} = \delta_3 \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{W}_3} = {\mathbf{X}_3}^T \cdot \delta_3
$$

$\large \frac{\partial \mathbf{J}}{\partial \mathbf{W}_2}$:


$\text{Now set }\delta_2 = \delta_3 \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{X}_3} \cdot \frac{\partial \mathbf{X}_3}{\partial \mathbf{Z}_2}$

Since $\mathbf{Z}_3 = \mathbf{X}_3 \cdot \mathbf{W}_3 \to \frac{\partial \mathbf{Z}_3}{{\partial \mathbf{X}_3}} = \mathbf{W}_3$ and $\mathbf{X}_3 = \text{ReLU}(\mathbf{Z}_2) \text{ so } \frac{\partial \mathbf{X}_3}{\partial \mathbf{Z}_2} = \text{ReLU}'(\mathbf{Z}_2)$ therefore:

$$
\delta_2 = \delta_3 \cdot \frac{\partial \mathbf{Z}_3}{\partial \mathbf{X}_3} \cdot \frac{\partial \mathbf{X}_3}{\partial \mathbf{Z}_2} = \delta_3 \cdot {\mathbf{W}_3}^T * \text{ReLU}'(\mathbf{Z}_2)
$$
*Note: * indicates element-wise multiplication.*

Now notice:
$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_2} = \delta_2 \cdot \frac{\partial \mathbf{Z}_2}{{\partial \mathbf{W}_2}} = {\mathbf{X}_2}^T \cdot \delta_2
$$

$\large \frac{\partial \mathbf{J}}{\partial \mathbf{W}_1}$:


$\text{Now set }\delta_1 = \delta_2 \cdot \frac{\partial \mathbf{Z}_2}{\partial \mathbf{X}_2} \cdot \frac{\partial \mathbf{X}_2}{\partial \mathbf{Z}_1} \to \delta_1 = \delta_2 \cdot {\mathbf{W}_2}^T * \text{ReLU}'(\mathbf{Z}_1)$

and notice:
$$
\frac{\partial \mathbf{J}}{\partial \mathbf{W}_1} = \delta_1 \cdot \frac{\partial \mathbf{Z}_1}{\partial \mathbf{W}_1} = {\mathbf{X}_1}^T \cdot \delta_1
$$