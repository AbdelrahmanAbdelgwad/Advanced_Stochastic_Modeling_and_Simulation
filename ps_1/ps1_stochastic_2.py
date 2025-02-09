
import numpy as np
import math
import matplotlib.pyplot as plt
from ps1_stochastic_1 import U, Inverse_Transform, sample_std_dev

global lambda_val
lambda_val = 2/3

def F_inverse(u):
    x = (-3/2) * np.log(abs(1-u))
    return x

def fx(x):
    return (2/math.sqrt(math.pi))*(math.sqrt(x))*(math.e**(-x)) 

def gx(x):
    return lambda_val * math.e ** (-lambda_val * x)


class AR:
    def __init__(self, uniform_generator, inverse_transform, c, seed):
        self.uniform_generator = uniform_generator
        self.inverse_transform = inverse_transform
        self.c = c
        self.seed = seed
    
    def generate(self):
        while True:
            y = self.inverse_transform.generate()
            u = self.uniform_generator.generate()
            if u <= (fx(y)/(self.c*gx(y))):
                x = y
                return x
            
    def generate_samples(self, n):
        samples = []
        for i in range(n):
            x = self.generate()
            samples.append(x)
        return np.array(samples)



if __name__ == "__main__":
    a = 39373
    m = 2147483647
    seed = 714
    n = 30_000
    c = (2/(lambda_val*math.sqrt(math.pi))) * (math.sqrt(1/(2*(1-lambda_val)))) * math.e**(1/2)
    print(c)
    uniform_generator = U(seed = seed, a=a, m=m)
    inverse_transform = Inverse_Transform(uniform_generator=uniform_generator, F_inverse=F_inverse)
    acceptance_rejection = AR(uniform_generator, inverse_transform, c, seed)

    acceptance_rejection_samples = acceptance_rejection.generate_samples(n=n)

    # plot the histogram
    plt.hist(acceptance_rejection_samples, bins=50)
    plt.title("Histogram of Generated Samples")
    plt.savefig("histogram_2.png")
    plt.show()
