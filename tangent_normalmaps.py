# This is slow and I'm not 100% sure I've done everything entirely corrrect
# but the tangent maps mostly work with blender (few visual bugs).

import numpy as np
import cv2
from PIL import Image

def create_normalmap(depthmap, scale = 0.2,
                     pre_blur = None, sobel_gradient = 3, post_blur = None,
                     invert=False):
    """Generates normalmaps.
    :param depthmap: depthmap that will be used to generate normalmap
    :param pre_blur: apply gaussian blur before taking gradient, -1 for disable, otherwise kernel size
    :param sobel_gradient: use Sobel gradient, None for regular gradient, otherwise kernel size
    :param post_blur: apply gaussian blur after taking gradient, -1 for disable, otherwise kernel size
    :param invert: depthmap will be inverted before calculating normalmap
    """

    normalmap = depthmap if invert else depthmap * (-1.0)
    normalmap = scale*normalmap / (256.0)
    
    if pre_blur is not None and pre_blur > 0:
        normalmap = cv2.GaussianBlur(normalmap, (pre_blur, pre_blur), pre_blur)

    # take gradients
    if sobel_gradient is not None and sobel_gradient > 0:
        zx = cv2.Sobel(np.float64(normalmap), cv2.CV_64F, 1, 0, ksize=sobel_gradient)
        zy = cv2.Sobel(np.float64(normalmap), cv2.CV_64F, 0, 1, ksize=sobel_gradient)
    else:
        zy, zx = np.gradient(normalmap)

    # combine and normalize gradients
    normal = np.dstack((zx, -zy, np.ones_like(normalmap)))
    # every pixel of a normal map is a normal vector, it should be a unit vector
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # TODO: this probably is not a good way to do it
    if post_blur is not None and post_blur > 0:
        normal = cv2.GaussianBlur(normal, (post_blur, post_blur), post_blur)
        # Normalize every vector again
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255, so we can export them
    normal += 1
    normal /= 2
    normal = np.clip(normal * 256, 0, 256 - 0.1)  # Clipping form above is needed to avoid overflowing
    normal = normal.astype(np.uint8)

    return Image.fromarray(normal)

# without projection math.
def create_tangent_normal(normal, depthmap, depthmap_resize, smooth_tangent = True):
    """Generates tangent normalmaps.
    :param normal: calculated the normal maps in object space (np.array)
    :param depthmap: depthmap that will be used to generate normalmap (np.array)
    :param depthmapsize: new size for depthmap. This should be a reduced amount ( (width, height) )
    """

    tangent_normals = np.array(normal)
    reduced_depth = np.array(Image.fromarray(depthmap).resize(depthmap_resize))
    # reduced_normal = np.array(Image.fromarray(normal).resize(depthmap_resize))

    scale_factor = 2
    depth_scale =  scale_factor*np.max(depthmap)

    reduced_depth_height_length = 1/ depthmap_resize[1]
    reduced_depth_width_length =  1/ depthmap_resize[0]
	#how to deal with non integers (is this right?)
    reduced_ratio_height = int(depthmap.shape[1] / depthmap_resize[1])
    reduced_ratio_width = int(depthmap.shape[0] / depthmap_resize[0])
    tbnMatrix_at_reduced_normal = np.zeros(( depthmap_resize[0], depthmap_resize[1], 3, 3))

    #four for loops is terrible... this should be 1-2 max
    for width_count in range(1,depthmap_resize[0]-1):
        for height_count in range(1,depthmap_resize[1]-1):

            pos1 = np.array([
            	-1.0 + 2 * width_count * reduced_depth_width_length,
            	-1.0 + 2 * height_count * reduced_depth_height_length,
            	reduced_depth[width_count-1][height_count]/depth_scale
            ])
            pos2 = np.array([
            	-1.0 + 2 * (width_count + 1) * reduced_depth_width_length,
            	-1.0 + 2 * height_count * reduced_depth_height_length,
            	reduced_depth[width_count + 1][height_count]/depth_scale
            ])
            pos3 = np.array([
            	-1.0 + 2 * width_count * reduced_depth_width_length,
            	-1.0 + 2 * height_count * reduced_depth_height_length,
            	reduced_depth[width_count][height_count-1]/depth_scale
            ])
            pos4 = np.array([
            	-1.0 + 2 * width_count * reduced_depth_width_length,
            	-1.0 + 2 * (height_count + 1) * reduced_depth_height_length,
            	reduced_depth[width_count][height_count+1]/depth_scale
            ])

            edge1 = (pos2 - pos1)
            edge2 = (pos4 - pos3)

            tangent = edge1/np.linalg.norm(edge1)
            cotangent = edge2/np.linalg.norm(edge2)
            
            normal_vec = np.cross(tangent, cotangent)
            bitangent = np.cross(tangent, normal_vec)
            
            tbnMat = np.array([tangent, bitangent, normal_vec])
            tbnMatrix_at_reduced_normal[width_count, height_count,:,:] = tbnMat

    for width_count in range(1, depthmap_resize[0]-1):
        for height_count in range(1, depthmap_resize[1]-1):
            reduced_map = normal[
                int(width_count*reduced_ratio_width):int((width_count+1)*reduced_ratio_width), 
                int(height_count*reduced_ratio_height):int((height_count+1)*reduced_ratio_height)
            ]/256
            # coordinate transform move origin to (0,0,0).
            reduced_map = 2*reduced_map - 0.99999
            reduced_map = reduced_map/np.linalg.norm(reduced_map)
        
            if smooth_tangent:
                point1 = 1/4 * (tbnMatrix_at_reduced_normal[width_count-1, height_count-1] + tbnMatrix_at_reduced_normal[width_count-1, height_count] + tbnMatrix_at_reduced_normal[width_count, height_count-1] + tbnMatrix_at_reduced_normal[width_count, height_count])
                point2 = 1/4 * (tbnMatrix_at_reduced_normal[width_count+1, height_count-1] + tbnMatrix_at_reduced_normal[width_count+1, height_count] + tbnMatrix_at_reduced_normal[width_count, height_count-1] + tbnMatrix_at_reduced_normal[width_count, height_count])
                point3 = 1/4 * (tbnMatrix_at_reduced_normal[width_count-1, height_count+1] + tbnMatrix_at_reduced_normal[width_count-1, height_count] + tbnMatrix_at_reduced_normal[width_count, height_count+1] + tbnMatrix_at_reduced_normal[width_count, height_count])
                point4 = 1/4 * (tbnMatrix_at_reduced_normal[width_count+1, height_count+1] + tbnMatrix_at_reduced_normal[width_count+1, height_count] + tbnMatrix_at_reduced_normal[width_count, height_count+1] + tbnMatrix_at_reduced_normal[width_count, height_count])
                    
            for i, array in enumerate(reduced_map):
                for j, vec in enumerate(array):
                    # # smooth tbn matrix..
                    if smooth_tangent:
                        width_weight = i / reduced_ratio_width
                        height_weight = j / reduced_ratio_height

                        tbn = (1-height_weight) * (1-width_weight) * point1 + (1-height_weight) * width_weight * point2 + height_weight * (1-width_weight) * point3 + height_weight * width_weight * point4
                    else:
                        tbn = tbnMatrix_at_reduced_normal[width_count, height_count] 

                    tbn = np.linalg.inv(tbn)
                    new_vec = np.matmul(tbn, vec)
                    new_vec = new_vec/np.linalg.norm(new_vec)
                    new_vec += 1
                    new_vec /= 2
                    new_vec = np.clip(new_vec * 256, 0, 256 - 0.1) 
                    
                    tangent_normals[width_count*reduced_ratio_width + i, height_count*reduced_ratio_height + j, :] = new_vec     
		    
    return (tangent_normals, reduced_depth)
