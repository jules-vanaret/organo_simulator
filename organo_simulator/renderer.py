import numpy as np
import stardist
from pyclesperanto_prototype import voronoi_labeling, select_device
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter as scipy_gaussian
from skimage.measure import regionprops


class Renderer:
    def __init__(self, nuclei_sizes, N_part, n_rays, Nx, d, L, cell_to_nuc_ratio,
                gaussian_blur_sigma, gaussian_noise_mean, gaussian_noise_sigma,
                scale=None):
        
        if isinstance(nuclei_sizes, int):
            nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part,1))
        elif isinstance(nuclei_sizes, np.ndarray):
            if nuclei_sizes.ndim == 1:
                nuclei_sizes = nuclei_sizes[:,None]  
        self.nuclei_sizes = nuclei_sizes
        self.max_nuclei_size = np.max(nuclei_sizes)

        self.n_rays = n_rays
        self.Nx = Nx
        self.d = d
        self.L = L
        self.cell_to_nuc_ratio = cell_to_nuc_ratio

        if scale is None:
            scale = (1,)*d
        self.scale = np.array(scale)

        if d==2:
            self.rendering_star_dist = lambda labels: stardist.geometry.geom2d.star_dist(labels, n_rays=n_rays, mode='opencl')
            self.rendering_polygons_to_label = stardist.geometry.geom2d.polygons_to_label
        elif d==3:
            rays = stardist.Rays_GoldenSpiral(n_rays)#, anisotropy=anisotropy)
            self.rendering_star_dist = lambda labels: stardist.geometry.geom3d.star_dist3D(labels,rays=rays, mode='opencl')
            self.rendering_polygons_to_label = lambda dist, points, shape: stardist.geometry.geom3d.polyhedron_to_label(dist, points, rays, shape, verbose=False)

        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_noise_mean = gaussian_noise_mean
        self.gaussian_noise_sigma = gaussian_noise_sigma

        # PyClEsperanto
        select_device(0)

        # import gputools
        # gputools.get_device().print_info()

    def make_labels_from_points(self, points):

        nuclei_sizes_pix = (self.nuclei_sizes*self.Nx/(2*self.L)).astype(int)

        prefactor_angles = np.pi if self.d==2 else 4/3*np.pi
        target_volume_pix = prefactor_angles*np.power(nuclei_sizes_pix, self.d)*self.cell_to_nuc_ratio

        prefactor_volume = prefactor_angles/self.n_rays


        centroid_mask = np.zeros((self.Nx,)*self.d, dtype=bool)
        coords_pix = tuple((points/(2*self.L)*self.Nx).T.astype(int))
        centroid_mask[coords_pix] = True

        voronoi_labels_t = np.array(voronoi_labeling(centroid_mask))
        #print(1)
        #voronoi_labels[ind_t] = voronoi_labels_t

        all_distances_sd = self.rendering_star_dist(voronoi_labels_t)    
        #print(2)    
        all_distances_sd = all_distances_sd[coords_pix] 
        
        all_distances = all_distances_sd.copy() 
        all_distances = np.clip(all_distances * (0.5*self.cell_to_nuc_ratio),0,nuclei_sizes_pix) / (0.5*self.cell_to_nuc_ratio)
        #all_distances = np.clip(all_distances ,0,nuclei_sizes_pix/0.4)*0.4


        problematic_distances_mask = np.less(all_distances, nuclei_sizes_pix)
        sum_prob_dist2 = np.sum(np.power(problematic_distances_mask*all_distances,self.d), axis=1)
        sum_ok_dist2 = np.sum(np.power((~problematic_distances_mask)*all_distances,self.d), axis=1)

        num = (target_volume_pix/prefactor_volume)[:,0] - sum_prob_dist2
        factors = np.power(
            num/sum_ok_dist2,
            1/self.d
        )

        multiplication_mask = problematic_distances_mask * 1 +(~problematic_distances_mask) * factors[:,None]

        all_distances = all_distances * multiplication_mask

        all_distances[sum_ok_dist2==0] = all_distances_sd[sum_ok_dist2==0]

        all_distances = np.clip(all_distances, 0, all_distances_sd)


        all_distances = all_distances * self.cell_to_nuc_ratio
        
        
        #print(3)
        labels = self.rendering_polygons_to_label(
            all_distances,
            (points/(2*self.L)*self.Nx).astype(int),
            (self.Nx,)*self.d
        )
        #print(4)
        return labels

    def make_realistic_data_from_labels(self, labels, debug=False):

        if debug:
            import napari
            viewer = napari.Viewer()
        
        labels_mask = labels.astype(bool)

        props = regionprops(labels)

        data = 0.005 * np.ones(labels.shape, dtype=float)
        if debug: viewer.add_image(data, scale=self.scale)



        for prop in props:
            data[prop.slice][prop.image] = np.random.normal(loc=0.07, scale=0.03)
        data = np.clip(data, 0,1)
        # data = labels_mask*1.0 * 0.1+0.01
        data = data + 0.005
        if debug: viewer.add_image(data, scale=self.scale)
        
        # Salt & pepper noise
        # data = random_noise(data, mode='s&p', clip=False, amount=0.1)  
        data_sp = random_noise(data, mode='s&p', clip=False, amount=0.1)  
        if debug: viewer.add_image(data_sp*data*10, scale=self.scale, name='spec')
        data = data + data_sp*data*10
        data[~labels_mask] = 0.005
        if debug: viewer.add_image(data, scale=self.scale, name='before')

        #Gaussian blur
        sigmas = [self.gaussian_blur_sigma/s for s in self.scale/self.scale.min()]
        data = scipy_gaussian(data, sigma=sigmas, )
        if debug: viewer.add_image(data, scale=self.scale,  name='after')

        # Gaussian noise
        data = random_noise(
            data,
            mode='gaussian',
            mean=self.gaussian_noise_mean,
            var=self.gaussian_noise_sigma**2,
            clip=False
        )
        data = np.clip(data, 0,1)
        if debug: viewer.add_image(data, scale=self.scale)
        # data = np.clip(data, 0,1)

        # Poisson noise
        data = random_noise(data, mode='poisson', clip=False)
        if debug: viewer.add_image(data, scale=self.scale)
        # data = np.clip(data, 0,1)


        

        # # Salt & pepper noise
        # data_sap = random_noise(data, mode='s&p', clip=False, amount=0.8)    
        # data[labels_mask] = data[labels_mask] + 0.1* 2*(data_sap[labels_mask]-1/2)
        if debug: napari.run()
        return data