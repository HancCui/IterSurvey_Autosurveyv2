from PIL import Image
import base64
import os
import io
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USE_WEB = True
SAMPLE_LIST = [] # 需要query的样本的unique_idx
MAX_RETRY = 3
TIME_OUT = 40
MAX_FILE_SIZE_MB = {'gpt-4o-2024-05-13':18,'gpt-4-turbo-2024-04-09': 18, 'gpt-4-1106-vision-preview': 18, 'gpt-4-vision-preview': 18, 'claude-3-opus-20240229': 3.5, 'gemini-pro-vision': 3, 'glm-4v-flash': 5}
EVAL_MODE = False

def check(a, b):
    return any(s in b for s in a)

def get_image_size(image_path):
    img = Image.open(image_path)
    return img.size

def get_image_format(image_path):
    with Image.open(image_path) as img:
        return img.format

def get_file_size_MB(file_path):
    file_size_bytes = os.path.getsize(file_path)
    file_size_MB = file_size_bytes / (1024 * 1024)  # Convert to MB
    return file_size_MB

def get_image_size_bytes(image, format):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format)  # Assuming the image is in JPEG format
    size_bytes = byte_arr.tell()  # Get the size of the io.BytesIO object
    return size_bytes / 1024 / 1024

def get_image_size_MB(image, format):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format, quality=85, optimize=True)
    size_bytes = byte_arr.tell()  # Get the size of the io.BytesIO object
    return size_bytes / 1024 / 1024

def scale_image_resolution(img, max_resolution):
    width, height = img.size
    if width > max_resolution or height > max_resolution:
        print('Image resolution exceeds the limit, resizing...')
        print(f'Image resolution: {img.size}')
        scale_factor = max(width, height) / max_resolution
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print('After resizing...')
        print(f'Image resolution: {img.size}')
    return img

def scale_image(img, file_size_MB, max_size_MB):
    """
    Scale the image to the max_size_MB, while maintaining the aspect ratio
    Args:
        img (PIL.Image.Image): The image to scale
        file_size_MB (float): The size of the image in MB
        max_size_MB (float): The maximum size of the image in MB

    Returns:
        PIL.Image.Image: The scaled image
    """
    print('Image size exceeds the limit, resizing...')
    print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')

    while file_size_MB > max_size_MB:
        # Calculate the scaling factor, while maintaining aspect ratio
        scale_factor = ((file_size_MB) / max_size_MB) ** 0.5

        # Resize the image
        width, height = img.size
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Check the new size
        file_size_MB = get_image_size_MB(img, format='jpeg')
        print('After resizing...')
        print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')
    return img


def encode_image(image):
    """
    Encode a PIL Image object to base64

    Args:
        image (PIL.Image.Image): The image to encode

    Returns:
        str: Base64 encoded string of the image
    """
    if image is None:
        print("Error: Cannot encode None image")
        return None

    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        img_bytes = buffer.read()
        if not img_bytes:
            print("Error: Empty image buffer")
            return None
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def convert_image_to_base64(image, max_size_MB, max_resolution=None):
    # 先检查image是否为None
    if image is None:
        print("Error: Image is None")
        return None

    try:
        # 1. 不再检查 .format，而是检查图像模式
        #    这是保存为JPEG的真正技术要求
        if image.mode not in ('RGB', 'L'): # L 是灰度模式，也可以保存为JPEG
            print(f"Image mode is '{image.mode}', converting to 'RGB' for JPEG compatibility.")
            image = image.convert('RGB')

        # 2. 如果图像太大，调整其大小 (您的原始逻辑)
        # 注意：此处get_image_size_MB的逻辑也需要是健壮的
        file_size_MB = get_image_size_MB(image, format='jpeg')

        if file_size_MB > max_size_MB:
            image = scale_image(image, file_size_MB, max_size_MB)

        # 3. 转换图像为base64
        base64_str = encode_image(image)
        return base64_str

    except Exception as e:
        print(f"Error processing image: {e}")
        return None





################################################################################
# Old version                                                                  #
################################################################################

# def scale_image(img, img_type, file_size_MB, max_size_MB):

#     print('Image size exceeds the limit, resizing...')
#     print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')

#     while file_size_MB > max_size_MB:
#         # Calculate the scaling factor, while maintaining aspect ratio
#         scale_factor = ((file_size_MB+1) / max_size_MB) ** 0.5

#         # Resize the image
#         width, height = img.size
#         new_width = int(width / scale_factor)
#         new_height = int(height / scale_factor)
#         img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

#         # Check the new size
#         file_size_MB = get_image_size_MB(img, format=img_type)
#         print('After resizing...')
#         print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')

#     # Save the resized image to a BytesIO object
#     byte_arr = io.BytesIO()
#     img.save(byte_arr, format=img_type)
#     size_bytes = byte_arr.tell()
#     byte_arr.seek(0)

#     return byte_arr, size_bytes / 1024 / 1024


# def scale_image_old(img, img_type, file_size_MB, max_size_MB):

#     print('Image size exceeds the limit, resizing...')
#     print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')

#     while file_size_MB > max_size_MB:
#         # Calculate the scaling factor, while maintaining aspect ratio
#         scale_factor = ((file_size_MB+1) / max_size_MB) ** 0.5

#         # Resize the image
#         width, height = img.size
#         new_width = int(width / scale_factor)
#         new_height = int(height / scale_factor)
#         img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

#         # Check the new size
#         file_size_MB = get_image_size_MB(img, format=img_type)
#         print('After resizing...')
#         print(f'Image resolution: {img.size}, Image size: {file_size_MB} MB')

#     # Save the resized image to a BytesIO object
#     byte_arr = io.BytesIO()
#     img.save(byte_arr, format=img_type)
#     size_bytes = byte_arr.tell()
#     byte_arr.seek(0)

#     return byte_arr, size_bytes / 1024 / 1024


# def convert_image_to_base64(image, max_size_MB, max_resolution=None):
#     img2format = {
#         'png': 'png',
#         'jpg': 'jpeg',
#         'jpeg': 'jpeg',
#         "PNG": "png",
#         "JPG": "jpeg",
#         "JPEG": "jpeg",
#         'webp': 'webp',
#     }
#     try:
#         img = image.convert('RGB')
#         img_type = image.format
#         if img_type not in img2format:
#             return None, None, None
#         img_type = img2format[img_type]
#     except Exception as e:
#         print(e)
#         return None, None, None

#     # Check the resolution of the image
#     if max_resolution is not None:
#         img = scale_image_resolution(img, max_resolution)

#     # Check the size of the image
#     file_size_MB = get_image_size_MB(img, format=img_type)
#     # print(f"Image size: {file_size_MB} MB")

#     if file_size_MB <= max_size_MB:
#         byte_arr = io.BytesIO()
#         img.save(byte_arr, format=img_type)
#         byte_arr.seek(0)
#     else:
#         byte_arr, file_size_MB = scale_image(img, img_type, file_size_MB, max_size_MB)

#     return byte_arr, img_type, file_size_MB

