import os
import numpy as np
from scipy.fftpack import dct, idct
from optparse import OptionParser
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class ImageDereflection:
    def __init__(self, gradient_thresh, epsilon):
        """
        Initializes an ImageDereflection object with the specified gradient threshold and epsilon value.

        Args:
            gradient_thresh (float): The threshold for suppressing gradients.
            epsilon (float): A small value to ensure unique solution and to 
                            prevent division by zero in the computations.

        Returns:
            An ImageDereflection object with the specified properties.
        """
        self.gradient_thresh = gradient_thresh
        self.epsilon = epsilon

    @staticmethod
    def gradient(image):
        """
            Calculate the gradient of the given image.

        Args:
            image (numpy.ndarray): A 2D array representing the input image.
            
        Returns:
            numpy.ndarray: A 3D array of shape (m, n, 2) representing the gradient of the input image.
        """
        m, n = image.shape
        gradient = np.zeros((m, n, 2))
        
        right_shifted = np.zeros((m, n))
        right_shifted[:, :n-1] = image[:, 1:n]
        right_shifted[:, n-1] = image[:, n-1]
        
        up_shifted = np.zeros((m, n))
        up_shifted[:m-1, :] = image[1:m, :]
        up_shifted[m-1, :] = image[m-1, :]
        
        gradient[:, :, 0] = right_shifted - image
        gradient[:, :, 1] = up_shifted - image
        
        return gradient

    @staticmethod
    def divergence(gradient):
        """
        Computes the divergence of a 2D gradient field.
        
        Args:
            gradient (numpy.ndarray): A 2D gradient field. The last dimension should have size 2.
        
        Returns:
            numpy.ndarray: The divergence of the input gradient field.
        """
        m, n, _ = gradient.shape
        divergence = np.zeros((m, n))
        
        T = gradient[:, :, 0]
        T1 = np.zeros((m, n))
        T1[:, 1:] = T[:, :n-1]
        
        divergence = divergence + T - T1
        
        T = gradient[:, :, 1]
        T1 = np.zeros((m, n))
        T1[1:, :] = T[:m-1, :]
        
        divergence = divergence + T - T1
        
        return divergence

    def solve_poisson_dct(self, rhs, mu, lam):
        """
        Solve the Poisson equation using the discrete cosine transform (DCT) method.
        
        Args:
            rhs (numpy.ndarray): The right-hand side of the Poisson equation to be solved.
            mu (float): A scalar parameter in the equation.
            lam (float): A scalar parameter in the equation.
        
        Returns:
            u (numpy.ndarray): The solution to the Poisson equation.
        """
        def dct2(block):
            return dct(dct(block.T, norm='ortho', type=2).T, norm='ortho', type=2)

        def idct2(block):
            return idct(idct(block.T, norm='ortho', type=2).T, norm='ortho', type=2)
        
        
        M, N = rhs.shape
        
        k = np.arange(1, M+1).reshape((1, M))
        l = np.arange(1, N+1).reshape((1, N))
        
        k = k.T
        
        eN = np.ones((1, N))
        eM = np.ones((M, 1))
        k = np.cos(np.pi/M*(k-1))
        l = np.cos(np.pi/N*(l-1))
        
        k = np.kron(k, eN)
        l = np.kron(eM, l)
        
        kappa = 2 * (k + l - 2)
        const = mu * kappa**2 - lam * kappa + self.epsilon
        u = dct2(rhs)
        
        u = u / const
        u = idct2(u)
        
        return u
        
    def dereflect(self, image):
        """
        Dereflects an input RGB image

        Args:
            image (numpy.ndarray): An RGB input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: A filtered RGB image with shape (height, width, channels), where the 
                            reflection is removed.
        """
        input_image = image.astype(np.float64)
        height, width, channels = input_image.shape
        filtered_image = np.zeros((height, width, channels))
        laplacian_pyramid = np.zeros((height, width, channels))
        
        for channel_idx in range(0, channels):
            gradient = self.gradient(input_image[:, :, channel_idx])
            gradient_x = gradient[:, :, 0]
            gradient_y = gradient[:, :, 1]
            gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_norm_thresh = np.where((gradient_norm <= self.gradient_thresh) & (gradient_norm >= -self.gradient_thresh), 0, gradient_norm)
            ind = (gradient_norm_thresh == 0)
            gradient_x[ind] = 0
            gradient_y[ind] = 0
            
            gradient_thresh = np.zeros((gradient_x.shape[0], gradient_x.shape[1], 2))
            gradient_thresh[:, :, 0] = gradient_x
            gradient_thresh[:, :, 1] = gradient_y
            laplacian_pyramid[:, :, channel_idx] = self.divergence(self.gradient(self.divergence(gradient_thresh)))
        
        filtered_image = laplacian_pyramid + self.epsilon * input_image
        
        for channel_idx in range(channels):
            filtered_image[:, :, channel_idx] = self.solve_poisson_dct(filtered_image[:, :, channel_idx], 1, 0)
            
        return filtered_image

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--thresh", dest="t", type='float', 
	                help="Default: %default", default=0.03)
    parser.add_option("-e", "--epsilon", dest="epsilon", type='float', 
	                help="Default: %default", default=1e-6)
    (options, args) = parser.parse_args()

    if len(args) == 1:
        assert(os.path.isfile(args[0]))
    elif len(args) == 2:
        assert os.path.isfile(args[0]) and os.path.isfile(args[1])
    else:
        print('Invalid usage.')
        exit()

    Im = cv2.cvtColor(cv2.imread(args[0]), cv2.COLOR_BGR2RGB)/255.0
    dereflect = ImageDereflection(options.t, options.epsilon)
    T = (dereflect.dereflect(Im)*255).astype(np.uint8)
    
    if len(args)>1:
        gt = cv2.cvtColor(cv2.imread(args[1]), cv2.COLOR_BGR2RGB)
        psnr = peak_signal_noise_ratio(gt,T)
        ssim = structural_similarity(gt,T,multichannel=True)
        print(f"PSNR = {psnr}")
        print(f"SSIM = {ssim}")

    cv2.imwrite(args[0].replace('.jpg', '_dereflected.jpg'), cv2.cvtColor(T, cv2.COLOR_RGB2BGR))