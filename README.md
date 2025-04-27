# comp9417-homework-1-solved
**TO GET THIS SOLUTION VISIT:** [COMP9417 Homework 1 Solved](https://www.ankitcodinghub.com/product/comp9417-machine-learning-solved-6/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;124054&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP9417 Homework 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
Tutorial: Regression I

In this tutorial we will explore the theoretical foundations of linear regression. We first work through linear regression in 1 dimension (univariate linear regression), and then extend the analysis to multivariate linear regression. If you find yourself unsure about any of the math covered in this tutorial, we strongly recommend reading the corresponding material in the text (freely available online):

Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong.

We will refer to this as the MML book throughout the course.

Question 1. (Univariate Least Squares)

A univariate linear regression model is a linear equation y = w0 + w1x. Learning such a model requires fitting it to a sample of training data, (x1,y1),…,(xn,yn), so as to minimize the loss (usually Mean

Squared Error (MSE)), . To find the best parameters w0 and w1 that minimize this error function we need to find the error gradients and . So we need to derive these expressions by taking partial derivatives, set them to zero, and solve for w0 and w1.

(a) Derive the least-squares estimates (minimizers of the MSE loss function) for the univariate linear regression model.

(b) Show that the centroid of the data, i.e. the point (x,y) is always on the least squares regression line.

(c) To make sure you understand the process, try to solve the following loss function for linear regression with a version of “L2” regularization, in which we add a penalty that penalizes the size of w1. Let λ &gt; 0 and consider the regularised loss

Question 2. (Multivariate Least Squares)

In the previous question, we found the least squares solution for the univariate (single feature) problem. We now generalise this for the case when we have p features. Let x1,x2,…,xn be n feature vectors (e.g.

corresponding to n instances) in Rp, that is:

 xi0  x

 i1  xi =  … 





xip−1

1

We stack these feature vectors into a single matrix, X ∈ Rn×p, called the design matrix. The convention is to stack the feature vectors so that each row of X corresponds to a particular instance, that is:

where the superscript T denotes the transpose operation. Note that it is standard to take the first element of the feature vectors to be 1 to account for the bias term, so we will assume xi0 = 1 for all i = 1,…,n. Analagously to the previous question, the goal is to learn a weight vector w ∈ Rp, and make predictions:

yˆi = wTxi = w0 + w1xi1 + w2xi2 + ··· + wp−1xi,p−1,

where yˆi denotes the i-th predicted value. To solve for the optimal weights in w, we can use the same procedure as before and use the MSE:

.

One approach to solve this would be to take derivatives with respect to each of the p weights and solve the resulting equations, but this would be extremely tedious. The more efficient way to solve this problem is to appeal to matrix notation. We can write the above loss as:

,

where k · k2 is the Euclidean norm. For the remainder of this question, we will assume that X is a full-rank matrix, which means that we are able to compute the inverse of XTX.

(a) Show that L(w) has a critical point:

wˆ = (XTX)−1XTy,

note: critical point here means that , i.e. the gradient evaluated at the critical point wˆ is zero).

Hint 1: if u is a vector in Rn, and v is a fixed vector in Rn, then .

Hint 2: if A is a fixed n × n matrix, and if f = zTAz, then .

(b) The condition is necessary but not sufficient to show that wˆ is a (global) minimizer of L, since this point could be a local minimum or a saddle point. Show that the critical point in part (a) is indeed a global minimizer of L.

Hint 1: L(w) is a function of w ∈ Rp+1, and so its Hessian, H, is the (p + 1) × (p + 1) matrix of

second order partial derivatives, that is, the (k,l)-th element of H is

.

Page 2

We will often write , where ∇ is the gradient operator, and ∇2 means taking the gradient twice. Note that the Hessian plays the role of second derivative for multivariate functions. Hint 2: a function is convex if its Hessian is positive semi-definite, which means that for any vector u,

uTHu ≥ 0.

Note also that this condition means that for any choice of u, the product term will always be nonnegative.

Hint 3: Any critical point of a convex function is a global minimum.

(c) In the next parts, we will use the formula derived above to verify our solution in the univariate case. We assume that p = 2, so that we have a two dimensional feature vector (one term for the intercept, and another for our feature). Write down the following values:

xi, y, w, X, XTX, (XTX)−1, XTy.

(d) Compute the least-squares estimate for the p = 2 case using the results from the previous part.

(e) Consider the following problem: we have inputs x1,…,x5 = 3,6,7,8,11 and outputs y1,…,y5 = 13,8,11,2,6. Compute the least-squares solution and plot the results both by hand and using python. Finally, use the sklearn implementation to check your results.

(g) Discuss the idea of a feature map. How would you use a feature map in the context of least squares regression?

(h) The mean-squared error (MSE) is

MSE ,

whereas the sum of squared errors (SSE) (also referred to as the residual sum of squares (RSS)) is

SSE .

Are the following statements True or False. Explain why.

(i) argminw∈Rp MSE(w) = argminw∈Rp SSE(w)

(ii) minw∈Rp MSE(w) = minw∈Rp SSE(w)

Notation: recall that minx g(x) is the smallest value of g(x), whereas argminx g(x) is the value of x that minimizes g(x). So minx(x − 2)2 = 0 but argminx(x − 2)2 = 2.

Question 4. (Population Versus Sample Parameters)

(a) What is the difference between a population and a sample?

(b) What is a population parameter? How can we estimate it?

Page 3
