import os

import OpenImageIO as oiio
import torch
from OpenImageIO import ImageBuf, ImageBufAlgo, ImageInput, ImageOutput, ImageSpec
from OpenImageIO.OpenImageIO import ROI

DEFAULT_INPUT_TRANSFORM = "scene_linear"
DEFAULT_OUTPUT_TRANSFORM = "sRGB"

# TODO: Add support for sublayers / arbitrary channels
# TODO: Support Display/View transforms? (probably not easy to adapt to comfy)
# TODO: Better error handling


class OIIO_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        colorspaces = OIIO_ColorspaceConvert.get_colorspaces()
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "precision": (["auto", "uint8", "half", "float"], {"default": "auto"}),
                "input_transform": (
                    colorspaces,
                    {"default": DEFAULT_INPUT_TRANSFORM},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("pixels", "xres", "yres")
    FUNCTION = "read"
    CATEGORY = "oiio"

    def read(self, path, precision, input_transform):
        img = ImageInput.open(path)
        if img:
            spec = img.spec()

            spec.attribute("oiio:ColorSpace", input_transform)

            precision = spec.format if precision == "auto" else precision
            xres = spec.width
            yres = spec.height
            nchannels = spec.nchannels

            pixels = img.read_image(0, 0, 0, nchannels, precision)

            img.close()
            pixels_tensor = torch.from_numpy(pixels)

            if pixels_tensor.dtype == torch.uint8:
                pixels_tensor = pixels_tensor.float() / 255.0

            if pixels_tensor.dim() == 3:
                pixels_tensor = pixels_tensor.unsqueeze(0)

            return (pixels_tensor, xres, yres)

        return (None, 0, 0)


class OIIO_ColorspaceConvert:
    @classmethod
    def get_colorspaces(cls):
        fallback = [
            "scene_linear",
            "lin_srgb",
            "lin_rec709",
            "Rec709",
            "sRGB",
            "linear",
            "ACEScg",
            "AdobeRGB",
            "KodakLog",
            "g22_rec709",
            "g18_rec709",
            "Gamma2.4",
        ]
        colorspaces = []
        try:
            config = oiio.ColorConfig()
            colorspaces = config.getColorSpaceNames()
        except Exception as e:
            print(f"Warning: Could not get colorspaces: {e}")

        return colorspaces if colorspaces else fallback

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (cls.get_colorspaces(), {"default": DEFAULT_INPUT_TRANSFORM}),
                "output_colorspace": (cls.get_colorspaces(), {"default": DEFAULT_OUTPUT_TRANSFORM}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "oiio"

    def convert(self, image, input_colorspace, output_colorspace):
        pixels = image.squeeze(0).cpu().numpy()
        spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], oiio.FLOAT)
        spec.attribute("oiio:ColorSpace", input_colorspace)

        in_buf = ImageBuf(spec)
        in_buf.set_pixels(ROI(), pixels)

        out_spec = oiio.ImageSpec(spec)
        out_spec.attribute("oiio:ColorSpace", output_colorspace)
        out_buf = oiio.ImageBuf(out_spec)

        # buf.set_write_format(oiio.FLOAT)
        # buf.copy_pixels(pixels)
        # buf.specmod().attribute("oiio:ColorSpace", input_colorspace)

        ImageBufAlgo.colorconvert(out_buf, in_buf, input_colorspace, output_colorspace)

        result = torch.from_numpy(out_buf.get_pixels()).unsqueeze(0)

        return (result,)


class OIIO_SaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        colorspaces = OIIO_ColorspaceConvert.get_colorspaces()
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"default": "output.exr"}),
                "precision": (["half", "float"], {"default": "half"}),
                "compression": (
                    ["none", "zip", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"],
                    {"default": "zip"},
                ),
                "colorspace": (
                    colorspaces,
                    {"default": DEFAULT_OUTPUT_TRANSFORM},
                ),
                "dwa_compression_level": ("FLOAT", {"default": 45.0, "min": 10.0, "max": 250.0}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "write"
    CATEGORY = "oiio"
    EXPERIMENTAL = True

    def write(self, images, path, precision, compression, colorspace, dwa_compression_level):
        base, ext = os.path.splitext(path)

        for i in range(images.shape[0]):
            # TODO: this isn't great as it would easily clash...
            output_path = path if images.shape[0] == 1 else f"{base}_{i:04d}{ext}"

            pixels = images[i].cpu().numpy()

            spec = ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], precision)
            spec.attribute("compression", compression)
            if compression in ["dwaa", "dwab"]:
                spec.attribute("compressionlevel", dwa_compression_level)

            spec.set_colorspace(colorspace)

            if pixels.shape[2] == 1:
                spec.channelnames = ("A",)
            elif pixels.shape[2] == 3:
                spec.channelnames = ("R", "G", "B")
            elif pixels.shape[2] == 4:
                spec.channelnames = ("R", "G", "B", "A")

            out = ImageOutput.create(output_path)
            if out:
                try:
                    out.open(output_path, spec)
                    out.write_image(pixels)
                finally:
                    out.close()

        return ()


NODE_CLASS_MAPPINGS = {
    "OIIO_LoadImage": OIIO_LoadImage,
    "OIIO_ColorspaceConvert": OIIO_ColorspaceConvert,
    "OIIO_SaveImage": OIIO_SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OIIO_ColorspaceConvert": "Colorspace Convert (OIIO)",
    "OIIO_LoadImage": "Load Image (OIIO)",
    "OIIO_SaveImage": "Save Image (OIIO)",
}
