N = total number of data point

Let's say we have 3 hidden layer 

X = 
activations function: 
g(x):
- including relu(), softmax()

Preprocess(X --> A(0)) --> layer 1(Z(1)->A(1)) --> layer 2(Z(2)->A(2)) 
--> layer 3(Z(3)->A(3)) == (Y: output)

A⁰∈(d × N) -> Z¹, A¹∈(c₁ × N) -> Z², A²∈(c₂ × N) -->  Z³, A³∈(C × N)

    layer 1        -->     layer 2        -->     layer 3        -> Ouput
  (c₁ × d)(d × N)  --> (c₂ × c₁)(c₁ × N)  --> (C × c₂)(c₂ x N)   -> (CxN)
W(1)^T A(0) + b(1) --> W(2)^T A(1) + b(2) --> W(3)^T A(2) + b(3) -> A(3)
Shape:
W ∈ c(l-₁) x c(l)


## Backpropagation:
depends on loss function:
    for cross-Entropy:
        L = -1/N * ∑(row=N)∑(column=feature) Y(r,c) log A(3)(r,c) 
    - Matrix Form
        L = -1/N * ∑(row=N) Y(row=N) log A(3)(row=N)

- When using softmax + cross-entropy, we do not need to compute:
    dL/dA(3) and dA(3)/dZ(3)
- because it's simply equal to:
    dL/dZ(3) = A(3) - Y (for single data-point)
    - Index notation
        dL/dZ(3) = 1/N ∑(row=N) (A(3) - Y) (for all data-point)
    - Matrix Form
        dL/dZ(3) = 1/N (A(3) - Y)

bar W(3):
dL/dW(3) = dL/dZ(3) * dZ(3)/dW(3)
dL/dW(3) = 1/N ∑(columns=N) (A(3)(c=dp) - Y(c=dp)) A(2)^T(c)
dL/dW(3) = 1/N (A(3) - Y) A(2)^T

bar b(3):
dL/db(3) = dL/dZ(3) * dZ(3)/db(3)
dL/db(3) = 1/N ∑(columns=N) (A(3)(c=dp) - Y(c=dp))

bar A(2), A(3-1) == A(l-1):
dL/dA(2) = dL/dZ(3) * dZ(3)/dA(2)
dL/dA(2) = 1/N (A(3) - Y) W(3)^T

bar W(2):
dL/dW(2) = dL/dZ(3) * dZ(3)/dA(2) * dA(2)/dW(2)
dL/dW(2) = 1/N (A(3) - Y) W(3)^T A(1)^T

bar b(2):
dL/db(2) = dL/dZ(3) * dZ(3)/dA(3) * dA(3)/db(2)
dL/db(2) = 1/N ∑(columns=N) (A(3)(c=dp) - Y(c=dp)) W(3)^T

bar A(1):
dL/dA(2) = dL/dZ(3) * dZ(3)/dA(3)




