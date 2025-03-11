import random as random_module
from PIL import Image, ImageDraw

def gen_random_image(width=512, height=512, shape_count=20):
    """
    Generate a random colored image with random shapes using an independent
    random number generator instance.
    
    Args:
        width (int): Width of the image
        height (int): Height of the image
        shape_count (int): Number of random shapes to draw
        
    Returns:
        PIL.Image: Generated random image
    """
    # Create a separate random number generator instance
    random_gen = random_module.Random()  # Uses its own independent state
    
    # Generate random background color
    background_color = (
        random_gen.randint(0, 255),
        random_gen.randint(0, 255),
        random_gen.randint(0, 255)
    )
    
    img_pil = Image.new('RGB', (width, height), color=background_color)
    
    # Draw some random colored shapes for visual interest
    draw = ImageDraw.Draw(img_pil)
    for i in range(shape_count):
        # Random position and size
        x1 = random_gen.randint(0, width)
        y1 = random_gen.randint(0, height)
        x2 = random_gen.randint(x1, min(x1 + 100, width))
        y2 = random_gen.randint(y1, min(y1 + 100, height))
        
        # Random color
        color = (
            random_gen.randint(0, 255),
            random_gen.randint(0, 255),
            random_gen.randint(0, 255)
        )
        
        # Draw ellipse
        draw.ellipse([x1, y1, x2, y2], fill=color)

    return img_pil