import random

from PIL import Image

from src import consts
from src.backgrounds import generate_background

MIN_CROP_SIZE = consts.BOARD_PIXEL_WIDTH + 10
MAX_CROP_SIZE = consts.BOARD_PIXEL_WIDTH * 4

TARGET_SIZE = consts.BOARD_PIXEL_WIDTH * 2


class NoboardGenerator:
    def generate_one(self) -> Image.Image:
        width = random.randint(MIN_CROP_SIZE, MAX_CROP_SIZE)
        height = random.randint(MIN_CROP_SIZE, MAX_CROP_SIZE)
        return generate_background(width, height).resize((TARGET_SIZE, TARGET_SIZE))
