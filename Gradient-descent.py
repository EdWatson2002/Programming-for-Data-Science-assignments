import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, gradf, x0, iterations=1000, eta=0.2):

    x_values = np.full_like([x0], x0)
    f_values = np.full_like([f(x0)], f(x0))
    gradf_values = np.full_like([gradf(x0)], gradf(x0))

    epsilon = 0.00000001

    for i in range(iterations):
        
        next_x = x_values[i] - eta*gradf_values[i]

        x_values = np.vstack([x_values, next_x])
        f_values = np.vstack([f_values, f(next_x)])
        gradf_values = np.vstack([gradf_values, gradf(next_x)])

        if np.linalg.norm(next_x - x_values[i]) < epsilon:
            return(np.squeeze(next_x), np.squeeze(x_values), 
                   np.squeeze(f_values), np.squeeze(gradf_values), i+1)
    
    print("Does not converge in time")
    

def gradient_ascent(f, gradf, x0, iterations=1000, eta=0.2):
    flipped_f = lambda x : -f(x)
    flipped_gradf = lambda x : -gradf(x)
    return(gradient_descent(flipped_f, flipped_gradf, x0, iterations, eta))

def momentum(f, gradf, x0, iterations=1000, eta=0.2, alpha = 0.9):
    
    x_values = np.vstack([np.full_like([x0], x0),x0])
    f_values = np.vstack([np.full_like([f(x0)], f(x0)),f(x0)])
    gradf_values = np.vstack([np.full_like([gradf(x0)], gradf(x0)),gradf(x0)])

    epsilon = 0.0000001

    for i in range(1,iterations):
        
        next_x = x_values[i] - eta*gradf_values[i] + alpha*(x_values[i] - x_values[i-1])
        x_values = np.vstack([x_values, next_x])
        f_values = np.vstack([f_values, f(next_x)])
        gradf_values = np.vstack([gradf_values, gradf(next_x)])

        if np.linalg.norm(next_x - x_values[i]) < epsilon:
            return(np.squeeze(next_x), np.squeeze(x_values), 
                   np.squeeze(f_values), np.squeeze(gradf_values), i)
    
    print("FAILED")
    return(np.squeeze(next_x), np.squeeze(x_values), 
                   np.squeeze(f_values), np.squeeze(gradf_values), i)

def f(x):
    return ((x[0]-1)**2 + 100*(x[0]**2 - x[1])**2)

def gradf(input):
    x = input[0]
    y = input[1]
    return(np.array([2*x - 2 + 400*x**3 - 400*x*y, -200*x**2 + 200*y]))

# f = lambda x : 4*(x**3)/3 + x**2 -3*x
# gradf = lambda x : 4*(x**2) + 2*x - 3

# f = lambda x : (x[0]-1)**2 + (x[1]-4)**2 - 3
# gradf = lambda x : np.array([2*x[0]-2, 2*x[1]-8])

results = momentum(f,gradf,np.array([3,4]), alpha=0.99, eta=0.0002, iterations=10000)
# results = gradient_descent(f,gradf,0)
print(results[4])
print(np.squeeze(results[1]))
# print(results[1])
# plt.plot(range(results[4]+1), results[1], label="x")
# plt.plot(range(results[4]+1), results[2], label="f(x)")
# plt.plot(range(results[4]+1), results[3], label="gradf(x)")
# plt.legend()
# plt.xlabel("Step number")
# plt.ylabel("Value")
# plt.grid(True)
# plt.show()
