######################################
# Modified code of part_utils.py
# Convert neural-renderer to pytorch3d 
# Modified by Hyeonwoo Kim
######################################
from pytorch3d.renderer.mesh.textures import TexturesUV
import torch
import numpy as np
import config
import pytorch3d
import os
import sys
sys.path.append(os.path.abspath(''))
os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
# Data structures and functions for rendering
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
    TexturesUV
)
from models import SMPL
#Input : pred_vertices, pred_camera
class PartRenderer():
    """Renderer used to render segmentation masks and part segmentations.
    Internally it uses the Neural 3D Mesh Renderer
    """
    def __init__(self, focal_length=5000., render_res=224):
        # Parameters for rendering
        self.focal_length = ((focal_length, focal_length),)
        self.render_res = render_res
        self.camera_center = ((render_res // 2, render_res // 2),)
        self.faces = torch.from_numpy(SMPL(config.SMPL_MODEL_DIR).faces.astype(np.int32)).cuda()
        textures = np.load(config.VERTEX_TEXTURE_FILE)
        self.textures = torch.from_numpy(textures).cuda().float() #float()
        self.cube_parts = torch.cuda.FloatTensor(np.load(config.CUBE_PARTS_FILE))

    def get_parts(self, parts, mask):
        """Process renderer part image to get body part indices."""
        bn,c,h,w = parts.shape
        mask = mask.view(-1,1)
        parts_index = torch.floor(100*parts.permute(0,2,3,1).contiguous().view(-1,3)).long()
        parts = self.cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None]
        parts *= mask
        parts = parts.view(bn,h,w).long()
        return parts

    def __call__(self, vertices, camera):
        """Wrapper function for rendering process."""
        device = torch.device('cuda')
        # Estimate camera parameters given a fixed focal length

        cam_t = torch.stack([camera[:,1], camera[:,2], 2*5000/(self.render_res * camera[:,0] +1e-9)],dim=0)
        batch_size = vertices.shape[0]
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)

        cameras = PerspectiveCameras(device = device, focal_length = self.focal_length, principal_point=self.camera_center, R = R, T = cam_t.reshape(batch_size, 3), image_size=((self.render_res, self.render_res),), in_ndc=False)

        raster_settings = RasterizationSettings(
            image_size=self.render_res, 
            blur_radius=0.0, 
        )

        lights = PointLights(device=device, location=[[5, 5, -5]]) 

        renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device = device,
                cameras=cameras,
                lights=lights
            )
        )
        
        faces_uvs = self.textures.transpose(2, 5)
        faces_uvs = faces_uvs[:, :, :, 0, 0, 0].expand(batch_size, -1, -1).to(device)

        vertices = vertices.reshape(vertices.shape[0], vertices.shape[1], -1).to(device)

        verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
 
        mesh = Meshes(
            verts = vertices,
            faces = faces_uvs,
            textures = textures)
        
        parts = renderer(mesh, cameras = cameras, lights = lights)
        rend_img = parts[:, :, :, :3] #[32, 224, 224, 3]
        mask = ~(rend_img == 1)[:,:,:,:,None]
        mask = mask.squeeze()
        mask = mask[:, :, :, :1]
        rend_img = torch.transpose(rend_img, 1, 3)
        rend_img = torch.transpose(rend_img, 2, 3)
        parts_ = self.get_parts(rend_img/255, mask)
        masks = torch.ones_like(mask)
        masks = masks * mask.int().float()
        return masks, parts_
