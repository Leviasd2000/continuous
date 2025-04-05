import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import scipy.special as sp

# Euler's method
# d\sigma^2(t) = -\lambda{\sigma^2(t)-\xi}dt+\omega\sigma(t)db(\lambda t)  where b is a brownian motion process
class SVProcess:
    def __init__(self, Lambda , xi , omega, sigma0 , M  ,  mu , s0 ,  jump_variance , h=1 , T=1 , rho=0 , jump=10 , jump_distribution = "Normal"): # Default: No Leverage Effect
        
        if xi >= (omega**2)/2 :
            raise ValueError("Error: xi must be < omega^2/2")
        if Lambda <= 0:
            raise ValueError("Error: lambda (l) must be > 0")
        
        self.T = T # time (day)
        self.dt = h/M # highincrement
        self.dm = h # lowincrement
        self.M = M # portions
        self.iteration = 100
        self.ds = h/(M*self.iteration)
        self.N = M*T*self.iteration
        
        """
        For CIR process
        """

        self.Lambda = Lambda
        self.xi = xi
        self.omega = omega
        self.sigma0 = sigma0
        
        """
        For SV process
        """
        self.mu = mu
        self.s0 = s0
        self.rho = rho # correlation between Wiener Process
        
        """
        For Jump process
        """
        self.step = jump
        self.variance = jump_variance
        self.distribution = jump_distribution
        
    def generatejump(self):
        jump2 = np.zeros(self.N)
        jump_positions = np.sort(np.random.choice(range(0, self.N), self.step, replace=False))
        if self.distribution == "Normal" :
            random_numbers = np.sqrt(self.variance)* np.random.randn(self.step)
            for m in range(self.step):
                for k in range(self.N):
                    if jump_positions[m]==k:
                        jump2[k]=random_numbers[m]
        return jump2
        
    def simulate(self, seed=None):
       
        if seed is not None:
            np.random.seed(seed)
        
        sigma2 = np.zeros(self.N)
        sigma2[0] = self.sigma0
        s2 = np.zeros(self.N)
        s2[0] = self.s0
        time_grid = np.linspace(0, self.T, self.N)

        for t in range(1, self.N):
            dWs = np.sqrt(self.ds) * np.random.randn()  # Wiener process 1
            dWp = self.rho * dWs + np.sqrt(1 - self.rho**2) * np.random.randn() * np.sqrt(self.ds) # Wiener process 2
            sigma2[t] = np.maximum(sigma2[t-1] + self.Lambda * (self.xi-sigma2[t-1]) * self.ds + self.omega * np.sqrt(sigma2[t-1]) * dWs,0) # nonnegativity
            s2[t] = s2[t-1]*np.exp((self.mu-0.5*sigma2[t-1])*self.ds + np.sqrt(sigma2[t-1])*dWp)

        return time_grid, sigma2 , s2  
    
    def RV_calculate(self,logstocklist):
        RV_list = np.zeros(int(len(logstocklist)/self.iteration)) 
        for k in range(len(RV_list)):
            if k==0:
                RV_list[k] = 0
            else:
                RV_list[k] = (logstocklist[k*self.iteration]-logstocklist[(k-1)*self.iteration])**2
        # sum of cumulative numbers
        cumulative_RV = np.array([np.sum(RV_list[i:i+self.M]) for i in range(0, len(RV_list), self.M)])
        return RV_list, cumulative_RV  
    
    def QV_calculate(self,sigmalist):
        QV_list = np.zeros(len(sigmalist))
        variable = 0
        for i in range(len(sigmalist)):
            variable = sigmalist[i]*(self.ds)
            QV_list[i] = variable
        # sum of cumulative numbers    
        cumulative_QV = np.zeros(len(QV_list))
        for i in range(self.M*self.iteration, len(QV_list)):
            cumulative_QV[i] = np.sum(QV_list[i-self.M*self.iteration+1:i+1])
        return QV_list, cumulative_QV
    
    def Expectation(self,degree):
        expectation = 2**( degree / 2) * (sp.gamma((degree + 1) / 2) / (np.sqrt(np.pi)))        
        return expectation
    
    def Bipower(self,logstocklist,degree_a,degree_b):
        Exp_a = self.Expectation(degree_a)
        Exp_b = self.Expectation(degree_b)
        power_list = np.zeros(int(len(logstocklist)/self.iteration)-1) 
        for k in range(len(power_list)):
            if k==0:
                power_list[k] = 0
            else:
                power_list[k] = (np.abs((logstocklist[k*self.iteration]-logstocklist[(k-1)*self.iteration]))**degree_a)*(np.abs((logstocklist[(k+1)*self.iteration]-logstocklist[k*self.iteration]))**degree_b)
        # sum of cumulative numbers
        cumulative_power = np.array([self.M**(1-(degree_a+degree_b)/2)/(Exp_a*Exp_b)*np.sum(power_list[i:(i+self.M-1)]) for i in range(0, len(power_list), self.M)])
        return power_list, cumulative_power  

            
    def plotstock(self, seed=None):
        for step in range(10):
            seed = step
            time, sigma2 , s2  = self.simulate(seed)
            # jump = self.generatejump()
            logs2 = np.log(s2) # exclude jump
            # logstock = logs2 + jump # include jump
            plt.plot(time, sigma2)
        plt.xlabel('Time')
        plt.ylabel('$sigma^2$')
        plt.title('CIR Square Root Process Simulation')
        plt.show()
        
    def QV_and_RVplot(self,RV,QV,Power,cumulative_RV, cumulative_QV,cumulative_power):
        time_gridRV = np.linspace(0, self.T, int(self.N/self.iteration))
        time_gridQV = np.linspace(0, self.T, self.N)
        print(self.dt)
        plt.plot(time_gridRV, RV, label="RV", color="blue")  
        plt.plot(time_gridQV, QV, label="QV", color="red")  
        plt.plot(time_gridQV, cumulative_QV , label="QV1" , color="green" )
        indices = np.linspace(1,self.T,self.T,endpoint=True)
        print(len(cumulative_RV))
        print(len(indices))
        plt.scatter(indices, cumulative_RV, color='blue', marker='x', label=f'Cumulative RV (every {self.M} steps)')
        plt.scatter(indices, cumulative_power, color='pink', marker='o', label=f'Cumulative Power (every {self.M} steps)')
        plt.legend()
        plt.title("RV and QV ")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    

# Test
if __name__ == "__main__":
    cir = SVProcess(Lambda=3,xi=0.2**2,omega=0.6,sigma0=0.25**2,M=72,T=50,mu=0.02,s0=1,jump_variance=0.64) # variance default 0
    s2 = cir.simulate(seed=10)[2]
    stock = np.log(s2)
    cir.plotstock(seed=10)
    sigma = cir.simulate(seed=10)[1]
    RV_list = cir.RV_calculate(stock)[0]
    QV_list = cir.QV_calculate(sigma)[0]
    power_list = cir.Bipower(stock,1,1)[0]
    RV_cum = cir.RV_calculate(stock)[1]
    QV_cum = cir.QV_calculate(sigma)[1]
    power_cum = cir.Bipower(stock,1,1)[0]
    power_cum = cir.Bipower(stock,1,1)[1]
    cir.QV_and_RVplot(RV_list,QV_list,power_list,RV_cum,QV_cum,power_cum)

    