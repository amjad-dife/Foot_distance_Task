import cv2      #  type: ignore
import torch # type: ignore
from torchvision.transforms import Compose # type: ignore
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet # type: ignore

def estimate_depth(frame, keypoints_2d):
    """
    Estimate depth for given 2D keypoints.
    Args:
        frame: Input frame (numpy array).
        keypoints_2d: 2D keypoints (e.g., ankle positions).
    Returns:
        keypoints_3d: 3D coordinates of the keypoints.
    """
    # Placeholder for depth estimation logic
    keypoints_3d = []
    # Replace this with actual depth estimation using Meta Sapien or MediaPipe Iris
    keypoints_3d = [(x, y, 0) for (x, y) in keypoints_2d]  # Default depth = 0


    return keypoints_3d 


def get_depth_map_using_depthAnyThing(image, depth_anything, device):
   
    transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),])

    # convert the image into RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = image / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device)
    depth = depth_anything(image) 

    # get the depth map 
    depth_map = depth[0].cpu().detach().numpy() 

    # normalize the depth map 
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0 

    print(f"depth_map shape : {depth_map.shape}")

    return depth_map

def get_depth_map_using_MiDas(image, midas, device):
    # Input transformation pipeline
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform 

        # Transform input for midas 
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    # Make a prediction
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        

        # # get the depth map 
        # depth_map = depth[0].cpu().detach().numpy() 

        # # normalize the depth map 
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0 

        # print(f"depth_map shape : {depth_map.shape}")

        return depth_map

def estimate_depth_anyThing(depth_map, keypoints_2d, original_frame_shape):
    """
    Estimate depth for given 2D keypoints.
    Args:
        depth_map: Depth map (numpy array).
        keypoints_2d: 2D keypoints (e.g., ankle positions).
        original_frame_shape: Shape of the original frame (height, width, channels).
    Returns:
        keypoints_3d: 3D coordinates of the keypoints.
    """
    original_height, original_width, _ = original_frame_shape
    depth_map_height, depth_map_width = depth_map.shape

    keypoints_3d = []

    for (x, y) in keypoints_2d:
        # Scale the keypoint coordinates to match the depth map size
        scaled_x = int(x * depth_map_width / original_width)
        scaled_y = int(y * depth_map_height / original_height)

        print(f"Original coordinates: ({x}, {y}), Scaled coordinates: ({scaled_x}, {scaled_y})")
        
        # Access the depth value from the depth map
        depth_value = depth_map[scaled_y, scaled_x]

        # Combine the 2D coordinates with the depth value
        keypoints_3d.append((x, y, depth_value))

    return keypoints_3d

def estimate_depth_MiDas(depth_map, keypoints_2d, original_frame_shape):
    """
    Estimate depth for given 2D keypoints.
    Args:
        depth_map: Depth map (numpy array).
        keypoints_2d: 2D keypoints (e.g., ankle positions).
        original_frame_shape: Shape of the original frame (height, width, channels).
    Returns:
        keypoints_3d: 3D coordinates of the keypoints.
    """
    original_height, original_width, _ = original_frame_shape
    depth_map_height, depth_map_width = depth_map.shape

    keypoints_3d = []

    for (x, y) in keypoints_2d:
        # Scale the keypoint coordinates to match the depth map size
        scaled_x = int(x * depth_map_width / original_width)
        scaled_y = int(y * depth_map_height / original_height)

        print(f"Original coordinates: ({x}, {y}), Scaled coordinates: ({scaled_x}, {scaled_y})")
        
        # Access the depth value from the depth map
        depth_value = depth_map[scaled_y, scaled_x]

        # Combine the 2D coordinates with the depth value
        keypoints_3d.append((x, y, depth_value))

    return keypoints_3d