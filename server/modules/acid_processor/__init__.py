import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# from .wobblers import WobbleMan
# from server.utils.segmentation_detection import HumanSeg
# from .infrared.colorize_infrared import ImageColorizationPipelineHF
from lunar_tools import resize, MedianBlur


def img2tensor(tensor):
    """
    Converts a tensor to a numpy array.

    Parameters:
    tensor (torch.Tensor): The input tensor to be converted.

    Returns:
    np.ndarray: The converted numpy array.
    """
    return (tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()) / 255.0


def tensor2image(input_data):
    """
    Converts a tensor to a numpy array.

    Parameters:
    input_data (torch.Tensor): The input tensor to be converted. It should be in the format (C, H, W) where
                                C is the number of channels, H is the height, and W is the width.

    Returns:
    np.ndarray: The converted numpy array or the input if it is not a tensor.
    """

    # Check if the input is a tensor
    if not isinstance(input_data, (torch.Tensor)):
        return input_data

    # Ensure the tensor is on the CPU and convert to a numpy array
    converted_data = input_data.cpu().numpy() if input_data.is_cuda else input_data.numpy()
    if len(converted_data.shape) == 4:
        converted_data = converted_data[0, :, :, :]
    converted_data = np.clip(converted_data * 255, 0, 255)
    return converted_data


def shear_image_torch(input_tensor, shear_factor_x=0.0, shear_factor_y=0.0):
    """
    Apply shearing transformation to a tensor image using affine grid.
    
    Parameters:
    input_tensor (torch.Tensor): The input image tensor with shape [H, W, C]
    shear_factor_x (float): Horizontal shear factor
    shear_factor_y (float): Vertical shear factor
    
    Returns:
    torch.Tensor: Sheared image tensor with the same shape as input
    """
    try:
        # print(f"[DEBUG] shear_factor_x={shear_factor_x}, shear_factor_y={shear_factor_y}")
        
        # Ensure the input is a 4D tensor [batch_size, channels, height, width]
        input_tensor = input_tensor.permute(2, 0, 1)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Get dimensions
        _, channels, height, width = input_tensor.shape
        
        # Create affine transformation matrix for shearing
        theta = torch.zeros((1, 2, 3), device=input_tensor.device)
        theta[0, 0, 0] = 1.0
        theta[0, 0, 1] = shear_factor_x
        theta[0, 1, 0] = shear_factor_y
        theta[0, 1, 1] = 1.0
        
        # Create sampling grid
        grid = F.affine_grid(theta, input_tensor.size(), align_corners=False)
        
        # Apply grid sampling
        sheared_tensor = F.grid_sample(input_tensor, grid, mode='bilinear', 
                                      padding_mode='reflection', align_corners=False)
        
        return sheared_tensor.squeeze(0).permute(1, 2, 0)  # Return to [H, W, C] format
    except Exception as e:
        print(f"shear_image_torch failed! {e}. returning original input")
        return input_tensor


def compress_image_torch(input_tensor, compression_factor_x=1.0, compression_factor_y=1.0):
    """
    Apply non-uniform compression to the image in x and/or y direction.
    
    Parameters:
    input_tensor (torch.Tensor): The input image tensor with shape [H, W, C]
    compression_factor_x (float): Horizontal compression factor (1.0 means no compression)
    compression_factor_y (float): Vertical compression factor (1.0 means no compression)
    
    Returns:
    torch.Tensor: Compressed image tensor with the same shape as input
    """
    try:
        print(f"[DEBUG] compression_factor_x={compression_factor_x}, compression_factor_y={compression_factor_y}")
        
        if compression_factor_x <= 0 or compression_factor_y <= 0:
            print("[DEBUG] Invalid compression factors. Must be positive.")
            return input_tensor
            
        # Ensure the input is a 4D tensor [batch_size, channels, height, width]
        input_tensor = input_tensor.permute(2, 0, 1)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Get dimensions
        _, channels, height, width = input_tensor.shape
        
        # Calculate new dimensions
        new_width = int(width * compression_factor_x)
        new_height = int(height * compression_factor_y)
        
        # First compress
        compressed_tensor = F.interpolate(
            input_tensor, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Then stretch back to original size
        result_tensor = F.interpolate(
            compressed_tensor,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        return result_tensor.squeeze(0).permute(1, 2, 0)  # Return to [H, W, C] format
    except Exception as e:
        print(f"compress_image_torch failed! {e}. returning original input")
        return input_tensor


def zoom_image_torch(input_tensor, zoom_factor):
    try:

        print(f"[DEBUG] zoom_factor={zoom_factor}")
        # Ensure the input is a 4D tensor [batch_size, channels, height, width]
        input_tensor = input_tensor.permute(2, 0, 1)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Original size
        original_height, original_width = input_tensor.shape[2], input_tensor.shape[3]

        # Calculate new size
        new_height = int(original_height * zoom_factor)
        new_width = int(original_width * zoom_factor)

        # Interpolate
        zoomed_tensor = F.interpolate(
            input_tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        # Calculate padding to match original size
        pad_height = (original_height - new_height) // 2
        pad_width = (original_width - new_width) // 2

        # Adjust for even dimensions to avoid negative padding
        pad_height_extra = original_height - new_height - 2 * pad_height
        pad_width_extra = original_width - new_width - 2 * pad_width

        # Pad to original size
        if zoom_factor < 1:
            zoomed_tensor = F.pad(
                zoomed_tensor,
                (
                    pad_width,
                    pad_width + pad_width_extra,
                    pad_height,
                    pad_height + pad_height_extra,
                ),
                "reflect",
                0,
            )
        else:
            # For zoom_factor > 1, center crop to original dimensions
            start_row = (zoomed_tensor.shape[2] - original_height) // 2
            start_col = (zoomed_tensor.shape[3] - original_width) // 2
            zoomed_tensor = zoomed_tensor[
                :,
                :,
                start_row : start_row + original_height,
                start_col : start_col + original_width,
            ]

        return zoomed_tensor.squeeze(0).permute(1, 2, 0)  # Remove batch dimension before returning
    except Exception as e:
        print(f"zoom_image_torch failed! {e}. returning original input")
        return input_tensor


# grid resampler
def torch_resample(tex, grid, padding_mode="reflection", mode="bilinear"):
    #    import pdb; pdb.set_trace()
    if len(tex.shape) == 3:  # add singleton to batch dim
        return F.grid_sample(
            tex.view((1,) + tex.shape),
            grid.view((1,) + grid.shape),
            padding_mode=padding_mode,
            mode=mode,
        )[
            0, :, :, :
        ].permute([1, 2, 0])
    elif len(tex.shape) == 4:
        return F.grid_sample(tex, grid.view((1,) + grid.shape), padding_mode=padding_mode, mode=mode)[0, :, :, :].permute([1, 2, 0])
    else:
        raise ValueError("torch_resample: bad input dims")


def torch_rotate(x, a):
    theta = torch.zeros((1, 2, 3)).cuda(x.device)

    theta[0, 0, 0] = np.cos(a)
    theta[0, 0, 1] = -np.sin(a)
    theta[0, 1, 0] = np.sin(a)
    theta[0, 1, 1] = np.cos(a)

    basegrid = F.affine_grid(theta, (1, 2, x.shape[1], x.shape[2]))[0, :, :, :]
    return torch_resample(x.unsqueeze(0), basegrid)


class InputImageProcessor:
    def __init__(
        self,
        do_human_seg=True,
        do_blur=False,
        blur_kernel=3,
        do_infrared_colorize=False,
        device="cuda",
    ):
        self.device = device
        self.brightness = 1.0
        self.saturization = 1.0
        self.hue_rotation_angle = 0
        self.blur = None
        self.blur_kernel = blur_kernel
        self.resizing_factor_humanseg = 0.4  # how much humanseg img is downscaled internally, makes things faster.

        # #  image colorization model for infrared images
        self.do_human_seg = do_human_seg

        # human body segmentation
        # if self.do_human_seg:
        #     self.human_seg = HumanSeg(
        #         resizing_factor=self.resizing_factor_humanseg, device=device, apply_smoothing=True, gaussian_kernel_size=9, gaussian_sigma=3
        #     )
        # else:

        # NOTE: hack because of lib import
        self.human_seg = None
        self.set_blur_size(self.blur_kernel)

        self.do_infrared_colorize = do_infrared_colorize
        self.do_blur = do_blur
        self.flip_axis = None

        self.list_history_frames = []

    def set_resizing_factor_humanseg(self, resizing_factor):
        self.resizing_factor_humanseg = resizing_factor
        self.human_seg.set_resizing_factor(resizing_factor)

    def set_brightness(self, brightness=1):
        self.brightness = brightness

    def set_saturization(self, saturization):
        self.saturization = saturization

    def set_hue_rotation(self, hue_rotation_angle=0):
        self.hue_rotation_angle = hue_rotation_angle

    def set_blur_size(self, blur_kernel):
        if blur_kernel != self.blur_kernel or not self.blur:
            self.blur = MedianBlur((blur_kernel, blur_kernel))

    def set_blur(self, do_blur=True):
        self.do_blur = do_blur

    def set_human_seg(self, do_human_seg=True):
        self.do_human_seg = do_human_seg

    def set_infrared_colorize(self, do_infrared_colorize=True):
        self.do_infrared_colorize = do_infrared_colorize

    def set_flip(self, do_flip, flip_axis=1):
        if do_flip:
            self.flip_axis = flip_axis
        else:
            self.flip_axis = None

    # @exception_handler
    def process(self, img):
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0)
            img = img.cpu().numpy()
            img = np.asarray(255 * img, dtype=np.uint8)

        if self.flip_axis is not None:
            img = np.flip(img, axis=self.flip_axis)

        if self.do_blur:
            img_torch = torch.from_numpy(img.copy()).to(self.device).float()
            img = self.blur(img_torch.permute([2, 0, 1])[None])[0].permute([1, 2, 0]).cpu().numpy()

        # human body segmentation mask
        if self.do_human_seg:
            human_seg_mask = self.human_seg.get_mask(img)
            img = self.human_seg.apply_mask(img)
        else:
            human_seg_mask = None

        # adjust brightness
        img = img.astype(np.float32)

        # if infrared, take mean of RGB channels and place it into red channel
        # the image can be then color-rotated with hue adjustments to fit the prompt color space
        if self.do_infrared_colorize:
            img = self.infrared_colorizer.process(img)

        # # time-averaging
        # self.list_history_frames.append(img)
        # if len(self.list_history_frames) > 10:
        #     self.list_history_frames = self.list_history_frames[1:]
        #     img = np.mean(np.stack(self.list_history_frames), axis=0)

        img *= self.brightness
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        # convert the image to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
        # adjust saturization
        img_hsv[:, :, 1] *= self.saturization

        # Rotate the hue
        # Hue is represented in OpenCV as a value from 0 to 180 instead of 0 to 360...
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + (self.hue_rotation_angle / 2)) % 180

        # clip the values to stay in valid range
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        # convert the image back to BGR
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if human_seg_mask is not None:
            human_seg_mask *= 255
            human_seg_mask = np.repeat(np.expand_dims(human_seg_mask, 2), 3, axis=2)
            human_seg_mask = human_seg_mask.astype(np.uint8)

        return img, human_seg_mask


class AcidProcessor:
    def __init__(self, device="cuda:0", height_diffusion=576, width_diffusion=1024):
        self.device = device
        self.last_diffusion_image_torch = None
        self.width_diffusion = width_diffusion
        self.height_diffusion = height_diffusion

        # self.wobbleman = WobbleMan(device=device)
        # self.wobbleman.init("a01")

        self.acid_strength = 0.05
        self.acid_strength_foreground = 0.01
        self.coef_noise = 0.15
        self.x_shift = 0
        self.y_shift = 0
        self.zoom_factor = 1
        self.rotation_angle = 0
        self.do_acid_tracers = False

        self.do_acid_wobblers = False
        self.do_flip_invariance = False
        self.wobbler_control_kwargs = {}
        self.flip_state = 0
        self.stereo_scaling_applied = False
        self.color_matching = 0.0
        
        # print(f"[AcidProcessor] Initialized with dimensions: height={height_diffusion}, width={width_diffusion}")
        
    def set_wobbler_control_kwargs(self, wobbler_control_kwargs):
        self.wobbler_control_kwargs = wobbler_control_kwargs

    def set_acid_strength(self, acid_strength):
        self.acid_strength = acid_strength
        # print(f"[AcidProcessor] Set acid_strength to {acid_strength}")

    def set_acid_strength_foreground(self, acid_strength_foreground):
        self.acid_strength_foreground = acid_strength_foreground
        # print(f"[AcidProcessor] Set acid_strength_foreground to {acid_strength_foreground}")

    def set_coef_noise(self, coef_noise):
        self.coef_noise = coef_noise
        # print(f"[AcidProcessor] Set coef_noise to {coef_noise}")

    def set_x_shift(self, x_shift):
        self.x_shift = x_shift
        # print(f"[AcidProcessor] Set x_shift to {x_shift}")

    def set_y_shift(self, y_shift):
        self.y_shift = y_shift
        # print(f"[AcidProcessor] Set y_shift to {y_shift}")

    def set_zoom_factor(self, zoom_factor):
        self.zoom_factor = zoom_factor
        print(f"[AcidProcessor] Set zoom_factor to {zoom_factor}")

    def set_rotation_angle(self, rotation_angle):
        self.rotation_angle = rotation_angle
        print(f"[AcidProcessor] Set rotation_angle to {rotation_angle}")

    def set_acid_tracers(self, do_acid_tracers):
        self.do_acid_tracers = do_acid_tracers
        print(f"[AcidProcessor] Set acid_tracers to {do_acid_tracers}")

    def set_human_seg(self, do_human_seg):
        self.do_human_seg = do_human_seg
        print(f"[AcidProcessor] Set do_human_seg to {do_human_seg}")

    def set_do_acid_wobblers(self, do_acid_wobblers):
        self.do_acid_wobblers = do_acid_wobblers
        print(f"[AcidProcessor] Set do_acid_wobblers to {do_acid_wobblers}")

    def set_flip_invariance(self, do_flip_invariance):
        self.do_flip_invariance = do_flip_invariance
        print(f"[AcidProcessor] Set do_flip_invariance to {do_flip_invariance}")

    def set_stereo_image(self, do_stereo_image=False):
        if not self.stereo_scaling_applied:
            self.height_diffusion = self.height_diffusion * 2
            self.stereo_scaling_applied = True
            print(f"[AcidProcessor] Applied stereo scaling. New height={self.height_diffusion}")

    def set_color_matching(self, color_matching):
        self.color_matching = color_matching
        print(f"[AcidProcessor] Set color_matching to {color_matching}")

    def multi_match_gpu(
        self,
        list_images,
        weights=None,
        simple=False,
        clip_max="auto",
        gpu=0,
        is_input_tensor=False,
    ):
        """
        Match colors of images according to weights.
        """
        from scipy import linalg

        if is_input_tensor:
            list_images_gpu = [img.clone().float() for img in list_images]
        else:
            list_images_gpu = [torch.from_numpy(img.copy()).float().cuda(gpu) for img in list_images]

        if clip_max == "auto":
            clip_max = 255 if list_images[0].max() > 16 else 1

        if weights is None:
            weights = [1] * len(list_images_gpu)
        weights = np.array(weights, dtype=np.float32) / sum(weights)
        assert len(weights) == len(list_images_gpu)
        # try:
        assert simple == False

        def cov_colors(img):
            a, b, c = img.size()
            img_reshaped = img.view(a * b, c)
            mu = torch.mean(img_reshaped, 0, keepdim=True)
            img_reshaped -= mu
            cov = torch.mm(img_reshaped.t(), img_reshaped) / img_reshaped.shape[0]
            return cov, mu

        covs = np.zeros((len(list_images_gpu), 3, 3), dtype=np.float32)
        mus = torch.zeros((len(list_images_gpu), 3)).float().cuda(gpu)
        mu_target = torch.zeros((1, 1, 3)).float().cuda(gpu)
        # cov_target = np.zeros((3,3), dtype=np.float32)
        for i, img in enumerate(list_images_gpu):
            cov, mu = cov_colors(img)
            mus[i, :] = mu
            covs[i, :, :] = cov.cpu().numpy()
            mu_target += mu * weights[i]

        cov_target = np.sum(weights.reshape(-1, 1, 1) * covs, 0)
        covs += np.eye(3, dtype=np.float32) * 1

        # inversion_fail = False
        try:
            sqrtK = linalg.sqrtm(cov_target)
            assert np.isnan(sqrtK.mean()) == False
        except Exception as e:
            # inversion_fail = True
            sqrtK = linalg.sqrtm(cov_target + np.random.rand(3, 3) * 0.01)
        list_images_new = []
        for i, img in enumerate(list_images_gpu):
            Ms = np.real(np.matmul(sqrtK, linalg.inv(linalg.sqrtm(covs[i]))))
            Ms = torch.from_numpy(Ms).float().cuda(gpu)
            # img_new = img - mus[i]
            img_new = torch.mm(img.view([img.shape[0] * img.shape[1], 3]), Ms.t())
            img_new = img_new.view([img.shape[0], img.shape[1], 3]) + mu_target

            img_new = torch.clamp(img_new, 0, clip_max)

            assert torch.isnan(img_new).max().item() == False
            if is_input_tensor:
                list_images_new.append(img_new)
            else:
                list_images_new.append(img_new.cpu().numpy())
        return list_images_new

    # @exception_handler
    def process(self, image_input, human_seg_mask=None):
        if isinstance(image_input, torch.Tensor):
            image_input = image_input.squeeze(0)
            image_input = image_input.cpu().numpy()
            image_input = np.asarray(255 * image_input, dtype=np.uint8)
        
        # print(f"[AcidProcessor] Processing image with shape: {image_input.shape}")
        
        if self.last_diffusion_image_torch is None:
            # print("[AcidProcessor] ERROR: last_diffusion_image_torch=None. returning original image...")
            return image_input
            
        last_diffusion_image_torch = self.last_diffusion_image_torch
        # print(f"[AcidProcessor] last_diffusion_image_torch shape: {last_diffusion_image_torch.shape}")
        
        width_diffusion = self.width_diffusion
        height_diffusion = self.height_diffusion
        # print(f"[AcidProcessor] Using dimensions: height={height_diffusion}, width={width_diffusion}")

        # acid transform
        # wobblers
        # if self.do_acid_wobblers:
        #     required_keys = ["amp", "frequency", "edge_amp"]
        #     wobbler_control_kwargs_are_good = all(key in self.wobbler_control_kwargs for key in required_keys)
        #     if not wobbler_control_kwargs_are_good:
        #         print(
        #             "Some keys are missing in wobbler_control_kwargs. Required keys are: ",
        #             required_keys,
        #         )
        #     else:
        #         resample_grid = self.wobbleman.do_acid(last_diffusion_image_torch, self.wobbler_control_kwargs)
        #         last_diffusion_image_torch = torch_resample(
        #             last_diffusion_image_torch.permute([2, 0, 1]),
        #             ((resample_grid * 2) - 1),
        #         ).float()

        # zoom
        if self.zoom_factor != 1 and self.zoom_factor > 0:
            # print(f"[AcidProcessor] Applying zoom with factor: {self.zoom_factor}")
            last_diffusion_image_torch = zoom_image_torch(last_diffusion_image_torch, self.zoom_factor)
            # print(f"[AcidProcessor] After zoom, shape is: {last_diffusion_image_torch.shape}")

        # rotations
        if abs(self.rotation_angle) > 0:
            # print(f"[AcidProcessor] Applying rotation with angle: {self.rotation_angle}")
            # Calculate padding to prevent cropping
            max_dim = max(last_diffusion_image_torch.shape[0], last_diffusion_image_torch.shape[1])
            padding = int(max_dim * (1 - np.cos(np.radians(abs(self.rotation_angle)))))
            padding = (padding, padding)
            # Pad the image
            last_diffusion_image_torch = transforms.Pad(padding=padding, padding_mode="reflect")(
                last_diffusion_image_torch.permute(2, 0, 1)
            )
            # Rotate using bilinear interpolation
            last_diffusion_image_torch = transforms.functional.rotate(
                last_diffusion_image_torch,
                angle=self.rotation_angle,
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
                expand=False,
            ).permute(1, 2, 0)
            # Crop back to original size from the center
            h, w = last_diffusion_image_torch.shape[:2]
            h_start = (h - height_diffusion) // 2
            w_start = (w - width_diffusion) // 2
            last_diffusion_image_torch = last_diffusion_image_torch[
                h_start : h_start + height_diffusion, w_start : w_start + width_diffusion
            ]
            # print(f"[AcidProcessor] After rotation, shape is: {last_diffusion_image_torch.shape}")

        # acid plane translations
        if self.x_shift != 0 or self.y_shift != 0:
            # print(f"[AcidProcessor] Applying shift: x={self.x_shift}, y={self.y_shift}")
            last_diffusion_image_torch = torch.roll(last_diffusion_image_torch, (self.y_shift, self.x_shift), (0, 1))

        img_input_torch = torch.from_numpy(image_input.copy()).to(self.device).float()
        # print(f"[AcidProcessor] img_input_torch shape: {img_input_torch.shape}")
        
        if img_input_torch.shape[0] != height_diffusion or img_input_torch.shape[1] != width_diffusion:
            # print(f"[AcidProcessor] Resizing input from {img_input_torch.shape} to {(height_diffusion, width_diffusion)}")
            img_input_torch = resize(
                img_input_torch.permute((2, 0, 1)),
                size=(height_diffusion, width_diffusion),
            ).permute((1, 2, 0))
            # print(f"[AcidProcessor] After resize, shape is: {img_input_torch.shape}")

        if self.do_acid_tracers and human_seg_mask is not None:
            # print(f"[AcidProcessor] Applying acid tracers with human seg mask shape: {human_seg_mask.shape}")
            if len(human_seg_mask.shape) == 3:
                human_seg_mask = human_seg_mask[:, :, 0] / 255
            human_seg_mask_resized = np.expand_dims(cv2.resize(human_seg_mask, (width_diffusion, height_diffusion)), 2)
            human_seg_mask_torch = torch.from_numpy(human_seg_mask_resized).to(self.device)
            img_input_torch_current = img_input_torch.clone()
            try:
                # print(f"[AcidProcessor] Combining images with acid_strength={self.acid_strength}")
                img_input_torch = (1.0 - self.acid_strength) * img_input_torch + self.acid_strength * last_diffusion_image_torch
                # print(f"[AcidProcessor] After first combination, shape is: {img_input_torch.shape}")
                
                img_input_torch = human_seg_mask_torch * img_input_torch_current + (1 - human_seg_mask_torch) * img_input_torch
                # print(f"[AcidProcessor] After mask application, shape is: {img_input_torch.shape}")
                
                img_input_torch = (
                    1.0 - self.acid_strength_foreground
                ) * img_input_torch + self.acid_strength_foreground * last_diffusion_image_torch
                # print(f"[AcidProcessor] After foreground effect, shape is: {img_input_torch.shape}")
            except RuntimeError as e:
                print(f"[AcidProcessor] ERROR during image combination: {e}")
                print(f"[AcidProcessor] img_input_torch shape: {img_input_torch.shape}")
                print(f"[AcidProcessor] last_diffusion_image_torch shape: {last_diffusion_image_torch.shape}")
                print(f"[AcidProcessor] human_seg_mask_torch shape: {human_seg_mask_torch.shape}")
                raise e
        else:
            try:
                # print(f"[AcidProcessor] Combining images with acid_strength={self.acid_strength}")
                img_input_torch = (1.0 - self.acid_strength) * img_input_torch + self.acid_strength * last_diffusion_image_torch
                # print(f"[AcidProcessor] After combination, shape is: {img_input_torch.shape}")
            except RuntimeError as e:
                print(f"[AcidProcessor] ERROR during image combination: {e}")
                print(f"[AcidProcessor] img_input_torch shape: {img_input_torch.shape}")
                print(f"[AcidProcessor] last_diffusion_image_torch shape: {last_diffusion_image_torch.shape}")
                raise e

        # additive noise
        if self.coef_noise > 0:
            torch.manual_seed(420)
            t_rand = (torch.rand(img_input_torch.shape, device=img_input_torch.device)[:, :, 0].unsqueeze(2) - 0.5) * self.coef_noise * 255
            img_input_torch += t_rand
        # Apply color matching if enabled
        if self.color_matching > 0.01:
            if human_seg_mask is not None:
                if human_seg_mask.ndim == 2:
                    human_seg_mask = np.expand_dims(human_seg_mask, axis=2)
                mask_torch = (
                    F.interpolate(
                        torch.from_numpy(human_seg_mask).to(self.device).float().permute(2, 0, 1).unsqueeze(0),
                        size=last_diffusion_image_torch.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                # Threshold to make mask binary (0 or 1)
                mask_torch = (mask_torch > 0.5).float()
                # Use binary mask by applying the mask directly
                last_diffusion_image_torch_masked = last_diffusion_image_torch * mask_torch
                img_input_torch, _ = self.multi_match_gpu(
                    [img_input_torch, last_diffusion_image_torch_masked],
                    weights=[1 - self.color_matching, self.color_matching],
                    clip_max="auto",
                    gpu=0,
                    is_input_tensor=True,
                )
            else:
                img_input_torch, _ = self.multi_match_gpu(
                    [img_input_torch, last_diffusion_image_torch],
                    weights=[1 - self.color_matching, self.color_matching],
                    clip_max="auto",
                    gpu=0,
                    is_input_tensor=True,
                )

        img_input = img_input_torch.cpu().numpy()
        
        # Ensure the output is properly formatted for PIL.Image.fromarray
        # - Must be uint8 type
        # - Clamp values between 0-255
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)
        
        # Ensure we have proper shape (H, W, 3)
        if img_input.shape != (height_diffusion, width_diffusion, 3):
            # print(f"[AcidProcessor] WARNING: Unexpected shape {img_input.shape}, reshaping to ({height_diffusion}, {width_diffusion}, 3)")
            img_input = img_input.reshape(height_diffusion, width_diffusion, 3)
            
        # print(f"[AcidProcessor] Final processed image shape: {img_input.shape}, dtype: {img_input.dtype}")
        return img_input

    def process_input(self, image_input, human_seg_mask=None):
        """
        Simplified version of process() that only applies zoom to the input image.
        
        Parameters:
        image_input (numpy.ndarray or torch.Tensor): The input image to process
        human_seg_mask (numpy.ndarray, optional): Human segmentation mask. Defaults to None.
        
        Returns:
        numpy.ndarray: Processed image with zoom applied
        """
        # Convert tensor to numpy array if needed
        if isinstance(image_input, torch.Tensor):
            image_input = image_input.squeeze(0)
            image_input = image_input.cpu().numpy()
            image_input = np.asarray(255 * image_input, dtype=np.uint8)
        
        # print(f"[AcidProcessor] Processing input image with shape: {image_input.shape}")
        
        # Convert to torch tensor
        img_input_torch = torch.from_numpy(image_input.copy()).to(self.device).float()
        
        # Apply zoom if needed
        if self.zoom_factor != 1 and self.zoom_factor > 0:
            # print(f"[AcidProcessor] Applying zoom with factor: {self.zoom_factor}")
            img_input_torch = zoom_image_torch(img_input_torch, self.zoom_factor)
            # print(f"[AcidProcessor] After zoom, shape is: {img_input_torch.shape}")

        # acid plane translations
        if self.x_shift != 0 or self.y_shift != 0:
            # print(f"[AcidProcessor] Applying shift: x={self.x_shift}, y={self.y_shift}")
            img_input_torch = torch.roll(img_input_torch, (self.y_shift, self.x_shift), (0, 1))
        
        # Convert back to numpy array
        img_input = img_input_torch.cpu().numpy()
        
        # Ensure the output is properly formatted
        img_input = np.clip(img_input, 0, 255).astype(np.uint8)
        
        # print(f"[AcidProcessor] Final processed image shape: {img_input.shape}, dtype: {img_input.dtype}")
        return img_input

    def update(self, img_diffusion):
        # Convert PIL Image to numpy array if needed
        if isinstance(img_diffusion, Image.Image):
            img_diffusion = np.array(img_diffusion)
            
        # print(f"[AcidProcessor] Updating diffusion image with shape: {img_diffusion.shape}")
        self.last_diffusion_image_torch = torch.from_numpy(img_diffusion).to(self.device, dtype=torch.float)
        # print(f"[AcidProcessor] Updated last_diffusion_image_torch shape: {self.last_diffusion_image_torch.shape}")


if __name__ == "__main__":
    acid_process = InputImageProcessor()
