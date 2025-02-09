import numpy as np
import matplotlib.pyplot as plt

class U:
    def __init__(self, seed, a, m):
        self.seed = seed
        self.x = self.seed
        self.a = a
        self.m = m

    def generate(self):
        self.x = (self.a * self.x) % self.m
        return self.x / self.m
    
    def generate_samples(self, n,seed):
        self.x = seed
        samples = []
        for _ in range(n):
            self.x = (self.a * self.x) % self.m
            samples.append(self.x)
        samples = np.array(samples) / self.m
        return samples
    

class Inverse_Transform:
    def __init__(self, uniform_generator, F_inverse):
        self.uniform = uniform_generator
        self.F_inverse = F_inverse

    def generate(self):
        u = self.uniform.generate()
        x = self.F_inverse(u)
        return x

    def generate_samples(self, n, seed=None):
        uniform_samples = self.uniform.generate_samples(n = n, seed=seed)
        samples = self.F_inverse(uniform_samples)
        return samples
    
def F_inverse(u):
    x = (-2 * np.log(1-u))**(1/2)
    return x

def sample_std_dev(samples):
    n = len(samples)
    sample_mean = np.mean(samples)
    sum_squared_diff = np.sum((samples - sample_mean)**2)
    sample_std_dev = (sum_squared_diff / (n-1))**(1/2)
    return sample_std_dev

if __name__=="__main__":
    a = 39373
    m = 2147483647
    x0 = 714
    uniform_generator = U(seed = x0, a=a, m=m)
    inverse_transform = Inverse_Transform(uniform_generator=uniform_generator, F_inverse=F_inverse)
    n = 30_000
    samples = inverse_transform.generate_samples(n=n, seed=x0)
    # plot the histogram
    plt.hist(samples, bins=50)
    plt.title("Histogram of Generated Samples")
    plt.savefig("histogram_1.png")
    plt.show()

    # calculate the mean (4 decimal places)
    mean = np.mean(samples)
    print(f"Mean: {mean:.4f}")

    # calculate the sample standard deviation (4 decimal places)
    std_dev = sample_std_dev(samples)
    print(f"Sample Standard Deviation: {std_dev:.4f}")

    




