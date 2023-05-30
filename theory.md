# Lasso with ADMM

## 1. Least Absolute Shrinkage and Selection Operator (Lasso)

In <span style="color: red; ">Lasso regression</span>, loss function is defined as follows:

$$
{\rm Loss}({\bf w}) = \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf w}||}_{1}
$$

, where 
${\bf x} = {(x_{1}, x_{2}, ..., x_{d})}^{{\rm T}} \in {\mathbb R}^{d}$, 
${\bf X} = ({\bf x}_{1}, {\bf x}_{2}, ..., {\bf x}_{n}) \in {\mathbb R}^{d \times n}$, 
${\bf w} = {(w_{1}, w_{2}, ..., w_{d})}^{{\rm T}} \in {\mathbb R}^{d}$, 
${\bf y} \in {\mathbb R}^{n}$ and 
${||{\bf w}||}_{1}$ is L**1**-reguralization term, not L2.

Lasso regression adopts L1-regularization term to make as many elements of ${\bf w}$ as possible equal to zero, while Ridge regression uses L2 to reduce the norm of ${\bf w}$.

Lasso regression has no analytical solution, since it has both a diffetentiable term ${||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2}$ and a non-differentiable term $||{\bf w}||_{1}$ using the same parameter ${\bf w}$. Thus, we need to minimize the loss iteratively. One of the solution methods is <span style="color: red; ">ADMM (Alternating Direction Method of Multipliers)</span>.

## 2. Intorodusing ADMM

In ADMM, we redefine the loss function as:

$$
{\rm Loss}({\bf w}, {\bf z}) = \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf z}||}_{1} \\
{\rm s.t.}\quad{\bf w} = {\bf z}
$$

The key idea here is to the main objective ${||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2}$ and the regularization term $\lambda {||{\bf z}||}_{1}$. Since these terms do not share the parameter ${\bf w}$ anymore, they can be evaluated by considering separate cases based on absolute values. However, if the constraint is satisfied strictly, the function is equivalent to the original one. Therefore, we need to impose it loosely.

## 3. The Augmented Lagrangian Method

How to impose the constraint? Here, we intoroduce <span style="color: red; ">The Augmented Lagrangian Method</span>, but before that, we show two methods and their pros and cons.

- Lagrange Multiplier Method
- Penalty Method

### 3-1. Lagrange Multiplier Method

The well-known method for maximizing or minimizing the certain function under the constraint. We define the following function:

$$
\begin{align}
{\mathcal L}({\bf x}, {\bf z}, {\bf {\gamma}}) &= {\rm Loss}({\bf w}, {\bf z}) + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) \nonumber \\
 &= \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf z}||}_{1} + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) \nonumber \\
\end{align}
$$

where ${\gamma}^{T} \in {\mathbb R}^{d}$ is the Lagrange multipliers. Taking partial derivatives with respect to ${\bf x}, {\bf z}$ and ${\gamma}$ and setting them to zero, we can find the extremum of the function. This is what we want.

This method is very straightforward, and the time to obtain a solution is short.　However, the constraint is satisfied strictly.

### 3-2. Penalty Method

This method regards the constraint as the penalty, so to reduce the penalty helps to minimizing the loss function, but it is necessary to make it zero.

$$
\begin{align}
{\mathcal P}({\bf x}, {\bf z}, \rho) &= {\rm Loss}({\bf w}, {\bf z}) + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
 &= \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf z}||}_{1} + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
\end{align}
$$

Here $\rho > 0$ is a tuning parameter. The bigger $\rho$ makes the constraint stricter. In this method, we increase $\rho$ from a small value incrementally and perform optimization at each step. We can handle the impact of reguralization, but this costs much time.

### 3-3. The Augmented Lagrangian Method

The method is an approach that combines the flexibility of the penalty and the rigor of the Lagrange multiplier. Here we renew the loss function like:

$$
\begin{align}
J({\bf w}, {\bf z}, {\bf \gamma}) &= {\rm Loss}({\bf w}, {\bf z}) + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
&= \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf z}||}_{1} + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
\end{align}
$$

In this method, ${\bf w}, {\bf z}$ and ${\bf \gamma}$ are updated iteratively:

$$
{\bf w}^{(0)} = {\bf 0},\quad {\bf z}^{(0)} = {\bf 0},\quad {\bf \gamma}^{(0)} = {\bf 0}
$$

$$
\begin{align}
{\bf w}^{(t+1)} &= \underset{\bf w}{{\rm argmin}} J({\bf w}^{(t)}, {\bf z}^{(t)}, {\bf \gamma}^{(t)}) \nonumber \\
{\bf z}^{(t+1)} &= \underset{\bf z}{{\rm argmin}} J({\bf w}^{(t)}, {\bf z}^{(t)}, {\bf \gamma}^{(t)}) \nonumber \\
{\bf \gamma}^{(t+1)} &= {\bf \gamma}^{(t)} + \rho ({\bf w}^{(t)} - {\bf z}^{(t)}) \nonumber \\
\end{align}
$$

where ${\bf w}^{(t)}, {\bf z}^{(t)}$ and ${\bf \gamma}^{(t)}$ are parameters at timestep $t$. Its convergence criterion consists of the following two conditions:

$$
\begin{align}
{\rm primary} &: {\bf w} - {\bf z} = {\bf 0} \nonumber \\
{\rm secondary} &: {\bf w}^{(t+1)} = {\bf w}^{(t)}, \quad {\bf z}^{(t+1)} = {\bf z}^{(t)} \nonumber \\
\end{align}
$$

The primary condition is literally the constraint, while the secondary one wants parameters to converge.

Let us zoom at the rule of update. Like the original Lagrange multiplier, we determine ${\bf w}^{(t+1)}$ by partial derivatives:

$$
\begin{align}
\frac{\partial J}{\partial {\bf w}} = -{\bf X}({\bf y} - {\bf X}^T {\bf w}^{(t)}) + {\bf \gamma}^{(t)} + {\bf \rho}({\bf w}^{(t)} - {\bf z}^{(t)}) &= 0 \nonumber \\
\therefore{\bf w}^{(t+1)} &= {({\bf X}{\bf X}^{T} + \rho {\bf I})}^{-1} ({\bf X}{\bf y} - {\bf \gamma}^{(t)} + \rho {\bf z}^{(t)}) \nonumber \\
\end{align}
$$

The function is non-differentiable with respect to ${\bf z}$ because of ${||{\bf z}||}_{1}$, but we can solve the problem analytically by dealing with each element of it. That is, we rewrite the function as:

$$
J({\bf w}^{(t)}, (z_{1}, z_{2}, ..., z_{l}, ..., z_{d}), {\bf \gamma}^{(t)}) = \sum_{l=1}^{d} (\frac{\rho}{2} {(z_{l} - {w}_{l}^{(t)})}^{2} + {\lambda}|z_{l}| - {\gamma}_{l} z_{l}) + {\rm Const.}
$$

where we consider terms not including ${\bf z}$ as constant values. Here, we can take derivatives with respect of $z_{l}$. We consider different cases of the derivative based on the sign of $z_{l}$:

$$
\frac{\partial J}{\partial z_{l}} =
\left\{
    \begin{array}{ll}
        {\rho} (z_{l} - {w}_{l}^{(t)}) + {\lambda} - {\gamma}_{l} = 0 & z_{l} > 0 \\
        (z_{l} - {w}_{l}^{(t)}) - {\lambda} - {\gamma}_{l} = 0 & z_{l} < 0 \\
        (z_{l} - {w}_{l}^{(t)}) + {\lambda}{\alpha} - {\gamma}_{l} = 0 & z_{l} = 0 \\
    \end{array}
\right.
$$

$$
\therefore z_{l} =
\left\{
    \begin{array}{ll}
        {w}_{l}^{(t)} + \frac{1}{\rho} ({\gamma}_{l} - {\lambda}) && z_{l} > 0 \\
        {w}_{l}^{(t)} + \frac{1}{\rho} ({\gamma}_{l} + {\lambda}) && z_{l} < 0 \\
        0 && z_{l} = 0 \\
    \end{array}
\right.
$$

We want to simlify the rule. The above equation also leads:

$$
\therefore {w}_{l}^{(t)} + \frac{1}{\rho}{\gamma}_{l}
\left\{
    \begin{array}{ll}
        > \frac{\gamma}{\rho} & z_{l} > 0 \\
        < -\frac{\gamma}{\rho} & z_{l} < 0 \\
        \in [-\frac{\gamma}{\rho}, \frac{\gamma}{\rho}] & z_{l} = 0 \\
    \end{array}
\right.
$$

where ${\alpha} \in [-1, 1]$ is determined to satisfy the equation.
<span style="color: red; ">The Soft-Threshoiding method</span>:

$$
\begin{align}
S_{\lambda}(x) &=
\left\{
    \begin{array}{ll}
        x - {\lambda} & x > \lambda \\
        0 & x \in [-\lambda, \lambda] \\
        x + {\lambda} & x < -\lambda \\
    \end{array}
\right. \nonumber \\
 &= {\rm sign}(x){\rm max}(0, |x| - \lambda) \nonumber \\
\end{align}
$$

makes the update of $z_{l}$ be written simply as:

$$
{\hat z}_{l}^{(t+1)} = S_{\frac{\lambda}{\rho}} (w_{l}^{(t)} + \frac{1}{\rho} {\gamma}_{l})
$$

In the case of ${\bf \gamma}$, the update rule has been already shown as:

$$
{\bf \gamma}^{(t+1)} = {\bf \gamma}^{(t)} + \rho ({\bf w}^{(t)} - {\bf z}^{(t)})
$$

What it means? The penalty method increases the penalty term $\frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2}$, intending to optimize the parameters. This method adopts <span style="color: red; ">the gradient ascent algorithm</span> and the above is its formula.

## 4. Summary

<span style="color: red; ">Lasso regression</span> uses the loss function:

$$
{\rm Loss}({\bf w}) = \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf w}||}_{1}
$$

to make as many elements of ${\bf w}$ as possible equal to zero but it is non-differentiable. One of the solution is <span style="color: red; ">The Augmented Lagrangian Method</span>, which inherits both of the property of Lagrange multiplier method and penalty method.

In this method, the loss function is renewed as:

$$
\begin{align}
J({\bf w}, {\bf z}, {\bf \gamma}) &= {\rm Loss}({\bf w}, {\bf z}) + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
&= \frac{1}{2} {||{\bf y} - {\bf X}^{T}{\bf w}||}_{2}^{2} + \lambda {||{\bf z}||}_{1} + {\bf {\gamma}}^{T}({\bf w} - {\bf z}) + \frac{\rho}{2}{||{\bf w} - {\bf z}||}_{2}^{2} \nonumber \\
\end{align}
$$

and each parameter is updated incrementally under the update rule:

$$
{\bf w}^{(0)} = {\bf 0},\quad {\bf z}^{(0)} = {\bf 0},\quad {\bf \gamma}^{(0)} = {\bf 0}
$$

$$
\begin{align}
{\bf w}^{(t+1)} &= {({\bf X}{\bf X}^{T} + \rho {\bf I})}^{-1} ({\bf X}{\bf y} - {\bf \gamma}^{(t)} + \rho {\bf z}^{(t)}) \nonumber \\
{\hat z}^{(t+1)} &= S_{\frac{\lambda}{\rho}} (w^{(t)} + \frac{1}{\rho} {\gamma}) \nonumber \\
{\bf \gamma}^{(t+1)} &= {\bf \gamma}^{(t)} + \rho ({\bf w}^{(t)} - {\bf z}^{(t)}) \nonumber \\
\end{align}
$$

where $S_{\frac{\lambda}{\rho}}$ is the element-wise <span style="color: red; "> soft-thresholding function</span>:

$$
S_{\lambda}(x) = {\rm sign}(x){\rm max}(0, |x| - \lambda)
$$

If the following two conditions are satisfied, it will stop iteration:

$$
\begin{align}
{\rm primary} &: {\bf w} - {\bf z} = {\bf 0} \nonumber \\
{\rm secondary} &: {\bf w}^{(t+1)} = {\bf w}^{(t)}, \quad {\bf z}^{(t+1)} = {\bf z}^{(t)} \nonumber \\
\end{align}
$$