import numpy as np, matplotlib.pyplot as plt
m, n, dim, iters = 10, 10, 2, 500
data = np.random.rand(500, dim)
W = np.random.rand(m, n, dim)
sigma0, lr0 = max(m,n)/2, 0.3
for t in range(iters):
    x = data[np.random.randint(0, len(data))]
    bmu = np.unravel_index(np.argmin(np.linalg.norm(W - x, axis=2)), (m,n))
    lr = lr0 * np.exp(-t/iters)
    sigma = sigma0 * np.exp(-t/(iters/np.log(sigma0)))
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(np.array([i,j]) - bmu)
            h = np.exp(-d*2 / (2*sigma*2))
            W[i,j] += lr * h * (x - W[i,j])
    if t % 100 == 0: print(f"Iteration {t}/{iters}")
plt.scatter(data[:,0], data[:,1], s=10, c='gray')
plt.scatter(W[:,:,0], W[:,:,1], c='red')
plt.title("Self Organizing Map (SOM)")
plt.show()