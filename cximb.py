import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from zernike import RZern


class PCA:
    def __init__(self, meanfile='mean.txt.gz', scalefile='scale.txt.gz', componentsfile='components.txt.gz'):
        '''
Construct an object that can transform a point in a principle component basis 
to the original basis from which it was defined. In practice, this is
essentially a container for the results of calculations performed by
scikit-learn. Specifically, we assume a standard centering/scaling of the
original data before the PCA was performed. As such, a (column) vector `p` in
PC space transforms to the original space as

    x = m + s * (R p)
    
where m and s are the column-wise mean and standard deviation of the data table
used to define the PCA, and R is a rotation matrix encoding the PC's themselves
in the original coordinate system.

Arguments:
    meanfile: text file holding the mean of each original feature
                (sklearn.preprocessing.StandardScaler.mean_)
    scalefile: text file holding the standard deviation of each original feature
                (sklearn.preprocessing.StandardScaler.scale_)
    componentfile: text file holding the principle components
                (sklearn.decomposition.PCA.components_), one PC per row
        '''
        self.mean = np.loadtxt(meanfile)
        self.scale = np.loadtxt(scalefile)
        self.components = np.loadtxt(componentsfile)
    def pc_to_orig(self, p):
        '''
Transform a vector `p` in the principle component basis encoded by this object
to the original feature space. Multiple vectors can be transformed at once by
arranging them as rows of a 2D array.
        '''
        return (p @ self.components) * self.scale + self.mean


class ZPmodel:
    def __init__(self, dim, n=30, xi=0.05):
        self.n = n
        self.dim = dim
        cart = RZern(n)
        ddx = np.linspace(-1, 1, dim)
        ddy = np.linspace(-1, 1, dim)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)
        self.cart = cart
        self.coeffs = np.zeros(cart.nk)
        
        # stuff to transform the model image back to real space
        x_inv = np.linspace(-1, 1, dim)
        y_inv = np.linspace(-1, 1, dim)
        X_inv, Y_inv = np.meshgrid(x_inv, y_inv, indexing='ij')
        Theta_inv = np.arctan2(Y_inv, X_inv)
        R_inv = np.sqrt(X_inv**2 + Y_inv**2)
        rr = [0.0]
        norm = 1./self.radial_distortion(1.0, xi)
        rrprime = [0.0]
        while True:
            rr.append(rr[-1] + 0.001) # magic spacing
            rrprime.append(self.radial_distortion(rr[-1], xi) * norm)
            if rrprime[-1] > 1.5: # 1.5>sqrt(2)
                break
        rr = np.array(rr)
        rrprime = np.array(rrprime)
        interp_inv = interpolate.interp1d(rr, rrprime, kind='cubic')
        R_new = interp_inv(R_inv)
        X_new = R_new*np.cos(Theta_inv)
        Y_new = R_new*np.sin(Theta_inv)
        self.x_inv = x_inv
        self.y_inv = y_inv
        self.X_new = X_new
        self.Y_new = Y_new

    def radial_distortion(self, z, c):
        return np.log(1 + z/c) + z
    
    def undistort(self, PhiZ):
        zern_interp = RegularGridInterpolator((self.x_inv, self.y_inv), PhiZ)
        model_im = zern_interp(np.array([self.X_new.flatten(), self.Y_new.flatten()]).T)
        model_im.shape = PhiZ.shape
        return model_im
    
    def make_image(self, undistort=True):
        PhiZ = self.cart.eval_grid(self.coeffs, matrix=True)
        if undistort:
            return self.undistort(PhiZ)
        else:
            return PhiZ
    
    def rotate(self, theta_deg):
        theta = np.radians(theta_deg)
        l = self.cart.mtab # prints l coeffs (see zernike polynomials in wikipedia, Noll index order)
        new = np.zeros(len(self.coeffs)) # creates array of rotated coeffs
        constant = 1 # keeps track of whether we add or subtract to go to the other index (Z_2_2 -> Z_2_-2)
        for i in range (self.coeffs.size): 
            if l[i] == 0: 
                new[i] = self.coeffs[i]
            else: 
                new[i] = np.sin(l[i] * theta) * self.coeffs[i + constant] + np.cos(l[i] * theta) * self.coeffs[i] # actual transformation
                if constant == 1: 
                    constant = constant - 2
                elif constant == -1: 
                    constant = constant + 2
        self.coeffs = new
        
    def reflect(self): 
        self.coeffs = np.where(self.cart.mtab>=0, self.coeffs, -self.coeffs)


class PCGen:
    def __init__(self, scalefile='pc_std_mad.txt.gz'):
        scales = np.loadtxt(scalefile)
        self.std = scales[:,0]
        self.mad = scales[:,1]
    def get_mad(self):
        return np.random.normal(loc=0.0, scale=self.mad)
    def get_std(self):
        return np.random.normal(loc=0.0, scale=self.std)
