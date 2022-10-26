import numpy as np


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
    

def repeat_along_t(array, repeat):
    return np.stack((array,) * repeat, axis=0)


if __name__ == '__main__':

    print(make_bounding_box((1,1)))