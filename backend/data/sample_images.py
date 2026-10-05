"""
Sample Medical Image Generator for Midtown Emergency Sheba.
Generates realistic ophthalmic and emergency pathology test images using PIL & NumPy.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_cases")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_normal_eye():
    size = (300, 300)
    img = Image.new("RGB", size, (245, 235, 230))
    draw = ImageDraw.Draw(img)

    # Sclera ellipse
    draw.ellipse([20, 50, 280, 250], fill=(240, 243, 246), outline=(210, 200, 190), width=3)
    
    # Very fine conjunctival vessels (few, healthy)
    for i in range(4):
        x1 = 30 + i * 20
        y1 = 120 + i * 15
        draw.arc([x1, y1, x1 + 50, y1 + 30], start=0, end=140, fill=(225, 180, 180), width=1)

    # Iris
    center = (150, 150)
    radius = 65
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(85, 55, 35), outline=(50, 30, 20), width=2)
    
    # Iris striations
    for angle in range(0, 360, 12):
        rad = math.radians(angle)
        x_start = center[0] + int(25 * math.cos(rad))
        y_start = center[1] + int(25 * math.sin(rad))
        x_end = center[0] + int(60 * math.cos(rad))
        y_end = center[1] + int(60 * math.sin(rad))
        draw.line([x_start, y_start, x_end, y_end], fill=(110, 75, 45), width=1)

    # Pupil
    p_radius = 24
    draw.ellipse([center[0]-p_radius, center[1]-p_radius, center[0]+p_radius, center[1]+p_radius], fill=(12, 12, 15))

    # Corneal light reflex (Purkinje highlight)
    draw.ellipse([center[0]-14, center[1]-18, center[0]-6, center[1]-10], fill=(255, 255, 255))
    
    # Eyelid contours
    draw.arc([10, 20, 290, 200], start=180, end=360, fill=(215, 160, 140), width=4)
    draw.arc([10, 100, 290, 270], start=0, end=180, fill=(215, 160, 140), width=3)
    
    path = os.path.join(OUTPUT_DIR, "normal_eye.jpg")
    img.save(path, quality=95)
    return path

def create_corneal_ulcer():
    size = (300, 300)
    img = Image.new("RGB", size, (240, 220, 215))
    draw = ImageDraw.Draw(img)

    # Injected sclera (ciliary injection & red eye)
    draw.ellipse([20, 50, 280, 250], fill=(245, 195, 195), outline=(220, 110, 110), width=3)

    # Intense engorged blood vessels around limbus (circumcorneal flush)
    for angle in range(0, 360, 18):
        rad = math.radians(angle)
        x_base = 150 + int(70 * math.cos(rad))
        y_base = 150 + int(70 * math.sin(rad))
        x_tip = 150 + int(115 * math.cos(rad))
        y_tip = 150 + int(115 * math.sin(rad))
        draw.line([x_base, y_base, x_tip, y_tip], fill=(185, 30, 30), width=2)
        draw.arc([x_base-10, y_base-10, x_base+10, y_base+10], start=0, end=180, fill=(200, 40, 40), width=1)

    # Iris
    center = (150, 150)
    radius = 65
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(70, 50, 35))

    # Pupil
    p_radius = 22
    draw.ellipse([center[0]-p_radius, center[1]-p_radius, center[0]+p_radius, center[1]+p_radius], fill=(10, 10, 12))

    # CORNEAL ULCER INFILTRATE (Whitish-yellow stromal infiltrate in the lower-central cornea)
    u_center = (140, 165)
    u_rad = 22
    # Infiltrate glow
    draw.ellipse([u_center[0]-u_rad-4, u_center[1]-u_rad-4, u_center[0]+u_rad+4, u_center[1]+u_rad+4], fill=(220, 230, 190, 180), outline=(240, 245, 200))
    # Ulcer core
    draw.ellipse([u_center[0]-u_rad, u_center[1]-u_rad, u_center[0]+u_rad, u_center[1]+u_rad], fill=(245, 250, 220), outline=(200, 180, 100), width=2)
    # Ulcer bed crater
    draw.ellipse([u_center[0]-10, u_center[1]-10, u_center[0]+10, u_center[1]+10], fill=(255, 255, 240))

    # Fluorescein greenish halo edge
    draw.arc([u_center[0]-u_rad-2, u_center[1]-u_rad-2, u_center[0]+u_rad+2, u_center[1]+u_rad+2], start=45, end=270, fill=(120, 220, 80), width=2)

    path = os.path.join(OUTPUT_DIR, "corneal_ulcer.jpg")
    img.save(path, quality=95)
    return path

def create_cataract():
    size = (300, 300)
    img = Image.new("RGB", size, (245, 235, 230))
    draw = ImageDraw.Draw(img)

    # Sclera
    draw.ellipse([20, 50, 280, 250], fill=(242, 244, 246), outline=(210, 205, 200), width=2)

    # Iris
    center = (150, 150)
    radius = 65
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(75, 45, 30), outline=(40, 25, 15), width=2)

    # Pupil with MATURE NUCLEAR CATARACT (Dense grayish-white opacification in pupillary aperture)
    p_radius = 26
    # Opaque lens
    draw.ellipse([center[0]-p_radius, center[1]-p_radius, center[0]+p_radius, center[1]+p_radius], fill=(210, 215, 220), outline=(170, 175, 180), width=2)
    # Nuclear density core (amber-brownish/pearly tint)
    draw.ellipse([center[0]-16, center[1]-16, center[0]+16, center[1]+16], fill=(225, 220, 195))
    draw.ellipse([center[0]-8, center[1]-8, center[0]+8, center[1]+8], fill=(240, 235, 210))

    # Cortical spokes (radial opacities)
    for a in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(a)
        x2 = center[0] + int(24 * math.cos(rad))
        y2 = center[1] + int(24 * math.sin(rad))
        draw.line([center[0], center[1], x2, y2], fill=(245, 245, 250), width=2)

    # Corneal highlight
    draw.ellipse([center[0]-18, center[1]-18, center[0]-10, center[1]-10], fill=(255, 255, 255))

    path = os.path.join(OUTPUT_DIR, "cataract.jpg")
    img.save(path, quality=95)
    return path

def create_diabetic_retinopathy():
    size = (300, 300)
    # Orange-red retinal fundus background
    img = Image.new("RGB", size, (185, 75, 30))
    draw = ImageDraw.Draw(img)

    # Retinal glow gradient
    center = (150, 150)
    for r in range(140, 40, -10):
        shade = int(185 + (140 - r) * 0.4)
        draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r], fill=(min(shade, 230), 85, 35))

    # Optic Disc (Nasal side: left-center)
    disc_center = (80, 150)
    draw.ellipse([disc_center[0]-24, disc_center[1]-28, disc_center[0]+24, disc_center[1]+28], fill=(250, 210, 140), outline=(240, 180, 100), width=2)
    # Optic cup
    draw.ellipse([disc_center[0]-12, disc_center[1]-14, disc_center[0]+12, disc_center[1]+14], fill=(255, 240, 190))

    # Macula & Fovea (Temporal side: center-right)
    macula_center = (195, 150)
    draw.ellipse([macula_center[0]-25, macula_center[1]-25, macula_center[0]+25, macula_center[1]+25], fill=(140, 50, 20))
    draw.ellipse([macula_center[0]-6, macula_center[1]-6, macula_center[0]+6, macula_center[1]+6], fill=(110, 35, 15))

    # Retinal blood vessels (Arcades branching from optic disc)
    draw.line([(80, 150), (110, 90), (160, 70), (220, 65), (270, 75)], fill=(120, 20, 15), width=4)
    draw.line([(80, 150), (110, 210), (160, 230), (220, 235), (270, 225)], fill=(120, 20, 15), width=4)
    draw.line([(110, 90), (140, 110)], fill=(130, 25, 20), width=2)
    draw.line([(110, 210), (140, 190)], fill=(130, 25, 20), width=2)

    # DIABETIC RETINOPATHY LESIONS:
    # 1. Microaneurysms & Dot-Blot Hemorrhages (dark red round dots)
    hemorrhages = [(150, 110), (170, 125), (145, 180), (180, 175), (210, 110), (235, 140), (130, 140), (220, 190)]
    for hx, hy in hemorrhages:
        draw.ellipse([hx-4, hy-4, hx+4, hy+4], fill=(90, 10, 10))
        draw.ellipse([hx+7, hy-2, hx+9, hy], fill=(85, 8, 8))

    # 2. Hard Exudates (bright yellowish lipid deposits near macula)
    exudates = [(180, 130), (185, 135), (190, 128), (215, 145), (220, 150), (210, 165), (205, 170)]
    for ex, ey in exudates:
        draw.ellipse([ex-3, ey-3, ex+3, ey+3], fill=(255, 245, 150))
        draw.point((ex+2, ey-1), fill=(255, 255, 200))

    # 3. Cotton Wool Spot (soft fluffy white infarct)
    draw.ellipse([140, 130, 155, 142], fill=(240, 230, 220))

    path = os.path.join(OUTPUT_DIR, "diabetic_retinopathy.jpg")
    img.save(path, quality=95)
    return path

def create_acute_conjunctivitis():
    size = (300, 300)
    img = Image.new("RGB", size, (240, 215, 210))
    draw = ImageDraw.Draw(img)

    # Sclera intensely injected with bright red arborizing superficial vessels
    draw.ellipse([20, 50, 280, 250], fill=(250, 190, 190), outline=(215, 80, 80), width=4)

    # Dense network of tortuous conjunctival vessels
    np.random.seed(42)
    for _ in range(25):
        x1 = np.random.randint(30, 270)
        y1 = np.random.randint(60, 240)
        # Avoid center iris area
        if (x1 - 150)**2 + (y1 - 150)**2 > 70**2:
            x2 = x1 + np.random.randint(-25, 25)
            y2 = y1 + np.random.randint(-25, 25)
            draw.line([x1, y1, x2, y2], fill=(200, 30, 30), width=np.random.randint(1, 3))

    # Mucopurulent discharge at medial and lateral canthi
    draw.ellipse([25, 145, 55, 165], fill=(235, 230, 180))
    draw.ellipse([250, 140, 275, 160], fill=(235, 230, 180))

    # Iris (clear, not involved - hallmark distinguishing from keratitis/uveitis)
    center = (150, 150)
    radius = 62
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(80, 50, 35))
    # Pupil (reactive, normal size)
    draw.ellipse([center[0]-22, center[1]-22, center[0]+22, center[1]+22], fill=(15, 15, 20))
    draw.ellipse([center[0]-12, center[1]-15, center[0]-6, center[1]-9], fill=(255, 255, 255))

    path = os.path.join(OUTPUT_DIR, "acute_conjunctivitis.jpg")
    img.save(path, quality=95)
    return path

def generate_all_samples():
    print("Generating medical test cases for Midtown Emergency Sheba...")
    paths = {
        "normal_eye": create_normal_eye(),
        "corneal_ulcer": create_corneal_ulcer(),
        "cataract": create_cataract(),
        "diabetic_retinopathy": create_diabetic_retinopathy(),
        "acute_conjunctivitis": create_acute_conjunctivitis()
    }
    print("Generated sample cases:", paths)
    return paths

if __name__ == "__main__":
    generate_all_samples()
