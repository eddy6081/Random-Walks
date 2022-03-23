"""
Author: Christopher Z. Eddy, eddych@oregonstate.edu
Purpose: Produce N-dimensional random walk with mean mu, std sigma
"""
import numpy as np
import matplotlib.pyplot as plt

class Gaussian_Random_Walk(object):
    def __init__(self, N=1000, L=100, dim=3, mu=None, sigma=None):
        """
        INPUTS
        ----------------------------------------
        N = integer, Number of trajectories to simulate

        L = integer, Length of each trajectory to simulate

        dim = integer, dimensionality of simulation

        mu = list of floats with length equal to dim, or single float to be
            casted to each dimension. Default behavior sets each dimension mean
            to zero. mu is the normally distributed average step size variable.

        sigma = list of floats with length equal to dim, or single float to be
            casted to each dimension. Default behavior sets each dimension mean
            to one. sigma is the normally distributed standard deviation of step
            size variable.

        """
        self.N = N
        self.L = L
        self.dim = dim
        if mu is None:
            self.mu = [0.]*dim
        else:
            if len(mu)<dim:
                self.mu = [mu]*dim
            else:
                assert len(mu)==dim, "length of 'mu' does not match dimensionality"
                self.mu = mu
        if sigma is None:
            self.sigma = [1.]*dim
        else:
            if len(sigma)<dim:
                self.sigma = [sigma]*dim
            else:
                assert len(sigma)==dim, "length of 'sigma' does not match dimensionality"
                self.sigma = sigma

    def generate_walk(self):
        """
        Generate Random Walks in N dimensions.
        """
        print("Simulating {}D Random Walk over {} trajectories of length {}...".format(self.dim, self.N, self.L))
        steps = np.zeros(shape=(self.N, self.L-1, self.dim))
        for d in range(self.dim):
            d_steps = self.mu[d] + np.random.normal(size=(self.N,self.L-1)) * self.sigma[d] #steps in dimension d
            steps[:,:,d] = d_steps

        positions = np.zeros(shape=(self.N, self.L, self.dim))
        positions[:,1:,:] = np.cumsum(steps,axis=1)
        return positions

    def calculate_MSD(self, positions):
        """
        Given a set of continuous positions (No missing data), calculate the
        mean square displacement using time lags to increase statistical power.
        """
        tau_diffs = []
        for tau in range(1,self.L):
            #reshape array so that we compute the steps between every "tau"th elements.
            #take the differences.
            diff_steps = positions[:,tau:,:] - positions[:,:-tau,:] #N x L-1 steps x dim
            #we want an (NxL-1) x dim array now.
            diff_steps = np.reshape(diff_steps, (self.N*diff_steps.shape[1],self.dim))
            #now store it.
            tau_diffs.append(diff_steps)
        #next, we'll square each element, and then take the mean.
        sq_tau_diffs = [x**2 for x in tau_diffs]
        #now we'll take the sum over each dimension. i.e. deltax^2 + deltay^2 + deltaz^2 +...
        sum_sq_tau_diffs = [np.sum(x,axis=1) for x in sq_tau_diffs]
        #next, we'll take the mean.
        msd = [0]+[np.mean(x) for x in sum_sq_tau_diffs]
        msd_error = [0] + [np.std(x)/np.sqrt(len(x)) for x in sum_sq_tau_diffs]
        lags = list(range(self.L))
        return msd, msd_error, lags

    def generate_plots(self, positions, lags, msd, msd_error):
        """
        Create the plots MSD and of several trajectories if the dimensionality
        is low enough.
        """
        #now we can plot if the dimensionality permits it.
        if self.dim<=3:
            if self.dim==3:
                #3D random walk.
                fig = plt.figure()
                #show first 5 trajectories
                ax1 = fig.add_subplot(121, projection='3d')
                for n in range(5):
                    ax1.plot(positions[n,:,0],positions[n,:,1],positions[n,:,2],'-')
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_zlabel("Z")
                ax1.set_title("Trajectories")
                #plot MSD.
                ax2 = fig.add_subplot(122)
                ax2.errorbar(lags, msd, msd_error, ls='--', marker='o')
                ax2.set_xlabel(r"$\tau$")
                ax2.set_ylabel(r"$\sigma^2$")
                ax2.set_title("Mean Square Displacement")
                plt.show()

            elif self.dim==2:
                #2D random walk
                fig,(ax1,ax2) = plt.subplots(1,2)
                #show first 5 trajectories
                for n in range(5):
                    ax1.plot(positions[n,:,0],positions[n,:,1],'-')
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_title("Trajectories")
                #plot MSD.
                ax2.errorbar(lags, msd, msd_error, ls='--', marker='o')
                ax2.set_xlabel(r"$\tau$")
                ax2.set_ylabel(r"$\sigma^2$")
                ax2.set_title("Mean Square Displacement")
                plt.show()

            else:
                #1D random walk.
                fig,(ax1,ax2) = plt.subplots(1,2)
                #show first 5 trajectories
                for n in range(5):
                    ax1.plot(list(range(self.L)),positions[n,:,0],'-')
                ax1.set_xlabel("Time")
                ax1.set_ylabel("X")
                ax1.set_title("Trajectories")
                #plot MSD.
                ax2.errorbar(lags, msd, msd_error, ls='--', marker='o')
                ax2.set_xlabel(r"$\tau$")
                ax2.set_ylabel(r"$\sigma^2$")
                ax2.set_title("Mean Square Displacement")
                plt.show()
        else:
            fig,ax=plt.subplots(1)
            ax1.errorbar(lags, msd, msd_error, ls='--', marker='o')
            ax1.set_xlabel(r"$\tau$")
            ax1.set_ylabel(r"$\sigma^2$")
            ax1.set_title("Mean Square Displacement")
            plt.show()

    def run_analysis(self):
        """
        Pull all the functions together.
        """
        positions = self.generate_walk()
        msd, msd_error, lags = self.calculate_MSD(positions)
        self.generate_plots(positions, lags, msd, msd_error)
    """
    Some Notes for self.
    Diffusion coefficient = sigma^2 / dimensionality
    slope = 2*d*D*t -> 2*sigma^2*t?
    t=1, M(t)=12, d=3, sigma=2 -> slope=12. So 2*d*sigma = slope?
    t=2, M(t)=24, d=3, sigma=2
    """

#Probably want to do a run-and-flick, run-and-tumble, Levi process,
#Persistent random walk.
