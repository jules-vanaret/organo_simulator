import numpy as np
import shutil
import os
import glob
import tqdm
import matplotlib.pyplot as plt
try:
    import napari
except ImportError:
    print("Napari is not installed. Please install it to use the display_particles_in_napari function.")
    napari = None



def random_2d_unit_vectors(N_part):
    phi = np.random.uniform(0,2*np.pi,size=N_part)
    x = np.cos( phi )
    y = np.sin( phi )

    return np.array([x,y]).T

def random_3d_unit_vectors(N_part):
    phi = np.random.uniform(0,2*np.pi,size=N_part)
    costheta = np.random.uniform(-1,1,size=N_part)

    theta = np.arccos( costheta )
    x = np.sin( theta ) * np.cos( phi )
    y = np.sin( theta ) * np.sin( phi )
    z = np.cos( theta )

    return np.array([x,y,z]).T

def make_bounding_box(bb_shape: tuple, bb_width: int = 1, 
                      world_array_shape: tuple = None,
                      top_left_corner_coords: tuple = None):

    if len(bb_shape) == 2:

        bounding_box = np.ones(bb_shape, dtype=int)
        bounding_box[bb_width:-bb_width, bb_width:-bb_width] = 0

    elif len(bb_shape) == 3:
        
        bounding_box = np.ones(bb_shape, dtype=int)
        
        # actually don't put 'bb_width' into this, so
        # that it produces a nice looking "cornered"
        # bounding box
        bounding_box[1:-1, 1:-1, 1:-1] = 0
        
        bounding_box[[0, -1], bb_width : -bb_width, bb_width : -bb_width] = 0
        bounding_box[bb_width : -bb_width, [0, -1], bb_width : -bb_width] = 0
        bounding_box[bb_width : -bb_width, bb_width : -bb_width, [0, -1]] = 0
        
    elif len(bb_shape) == 4:

        bounding_box = make_bounding_box(bb_shape[1:], bb_width)
        bounding_box = repeat_along_t(bounding_box, repeat=bb_shape[0])
        
    else:
        print(f'Given shape has length {len(bb_shape)}')
        raise NotImplementedError

    if world_array_shape is not None:

        assert len(world_array_shape) == len(bb_shape)

        world_array = np.zeros(shape=world_array_shape, dtype=int)

        world_array[
            ...,            
            top_left_corner_coords[0] : top_left_corner_coords[0] + bb_shape[1],
            top_left_corner_coords[1] : top_left_corner_coords[1] + bb_shape[2],
            top_left_corner_coords[2] : top_left_corner_coords[2] + bb_shape[3]
        ] = bounding_box

        return world_array
    
    
    return bounding_box


def load_csv_coords(path_to_csv: str):
    return np.loadtxt(fname=path_to_csv, delimiter=',')
    

def repeat_along_t(array, repeat):
    return np.stack((array,) * repeat, axis=0)

def delete_content_of_dir(path_to_dir: str, content_type: str = ''):

    files = glob.glob(f'{path_to_dir}/*{content_type}')
    
    print(f'Deleting content of folder {path_to_dir}')
    if len(files)>0:
    
        for f in tqdm(files):
            try:
                os.remove(f)
            except IsADirectoryError:
                shutil.rmtree(f)

def display_particles_in_napari(tracks, nuclei_sizes, Nt, L):


    # random_ids = np.zeros(tracks.shape[0])
    unique_ids = np.unique(tracks[:,0])
    random_ids = np.random.choice(unique_ids, len(unique_ids), replace=False)

    all_random_ids = np.zeros(tracks.shape[0])
    for unique_id in unique_ids:
        all_random_ids[tracks[:,0] == unique_id] = random_ids[unique_ids == unique_id]

    props = {'random_ids': all_random_ids}



    viewer = napari.Viewer()
    tracks_layer = viewer.add_tracks(tracks, tail_length=6, blending='opaque', tail_width=8, name='tracks', properties=props)
    indices_tp = np.where(
        np.max(
            np.abs(np.diff(tracks_layer.data[:,2:], axis=0)),
            axis=1
        ) > L/2
    )
    tracks_layer._manager._track_connex[indices_tp] = False
    tracks_layer.events.rebuild_tracks()

    points =  tracks_layer.data[:,1:]
    sizes = np.repeat(nuclei_sizes, Nt, axis=0)
    face_colors_at_t = plt.cm.inferno(nuclei_sizes/np.max(nuclei_sizes))
    face_colors = np.repeat(face_colors_at_t, Nt, axis=0)

    viewer.add_points(points, size=sizes, face_color=face_colors, edge_color='black', name='particles')

    napari.run()