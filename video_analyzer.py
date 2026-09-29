import cv2
import numpy as np
import os
import random

import torch
import torchvision.transforms as transforms
from torchvision import models

import insightface
from insightface.app import FaceAnalysis

from ultralytics import YOLO
from nudenet import NudeDetector
import easyocr

# ──────────────────────────────────────────────────────────────────────
# 1. InsightFace — Face Detection & Embedding
# ──────────────────────────────────────────────────────────────────────
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

MAX_FACES_PER_FRAME = 5
INTRA_VIDEO_SIMILARITY_THRESHOLD = 0.6

# ──────────────────────────────────────────────────────────────────────
# 2. MobileNetV3 — Scene/Content Classification
# ──────────────────────────────────────────────────────────────────────
scene_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
scene_model.eval()

scene_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_imagenet_labels = None
def get_imagenet_labels():
    global _imagenet_labels
    if _imagenet_labels is None:
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            _imagenet_labels = weights.meta["categories"]
        except Exception:
            _imagenet_labels = []
    return _imagenet_labels

# ──────────────────────────────────────────────────────────────────────
# 2B. YOLOv8 — Object Detection & Pose Estimation
# ──────────────────────────────────────────────────────────────────────
yolo_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

# ──────────────────────────────────────────────────────────────────────
# 2C. NudeNet — NSFW Detection
# ──────────────────────────────────────────────────────────────────────
nude_detector = NudeDetector()

# ──────────────────────────────────────────────────────────────────────
# 2D. EasyOCR — Text Extraction
# ──────────────────────────────────────────────────────────────────────
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Mapping from ImageNet class substrings to content categories + tags
SCENE_CATEGORY_MAP = [
    # Nature & Landscapes
    (["cliff", "valley", "volcano", "lakeside", "seashore", "beach", "coral reef",
      "alp", "mountain", "geyser", "promontory", "sandbar"], "Nature", "nature"),
    (["ocean", "sea", "water", "river", "fountain", "dam"], "Nature", "water"),
    (["forest", "tree", "jungle", "rainforest", "wood"], "Nature", "forest"),
    (["sunset", "sunrise", "sky", "cloud", "star"], "Nature", "sky"),
    (["flower", "daisy", "rose", "tulip", "sunflower", "poppy", "lily",
      "bouquet", "pot", "vase"], "Nature", "flowers"),
    # Animals
    (["dog", "puppy", "retriever", "shepherd", "terrier", "poodle", "bulldog",
      "husky", "labrador", "beagle", "collie", "corgi", "pug", "hound",
      "chihuahua", "dalmatian", "schnauzer", "maltese"], "Animals", "dogs"),
    (["cat", "kitten", "tabby", "siamese", "persian", "tiger cat",
      "Egyptian cat"], "Animals", "cats"),
    (["bird", "parrot", "eagle", "owl", "flamingo", "peacock", "robin",
      "hummingbird", "toucan", "pelican", "crane"], "Animals", "birds"),
    (["fish", "shark", "whale", "dolphin", "jellyfish", "stingray",
      "goldfish", "clownfish", "lionfish", "starfish"], "Animals", "marine-life"),
    (["horse", "zebra", "donkey", "pony", "stallion"], "Animals", "horses"),
    (["lion", "tiger", "leopard", "cheetah", "jaguar", "panther", "cougar",
      "lynx", "bear", "panda", "polar bear", "elephant", "giraffe",
      "hippopotamus", "rhinoceros", "gorilla", "monkey", "chimpanzee",
      "orangutan", "baboon", "deer", "elk", "moose", "bison", "buffalo",
      "wolf", "fox", "coyote", "raccoon", "rabbit", "squirrel",
      "hamster", "guinea pig", "mouse", "rat", "hedgehog", "koala",
      "kangaroo", "sloth", "armadillo", "bat", "otter",
      "seal", "walrus", "camel", "llama", "alpaca"], "Animals", "wildlife"),
    (["snake", "lizard", "iguana", "chameleon", "gecko", "turtle",
      "tortoise", "crocodile", "alligator", "frog", "toad",
      "salamander", "newt"], "Animals", "reptiles"),
    (["butterfly", "moth", "bee", "wasp", "ant", "beetle", "ladybug",
      "dragonfly", "grasshopper", "cricket", "spider", "scorpion",
      "caterpillar", "centipede", "millipede", "snail", "slug",
      "worm"], "Animals", "insects"),
    # Food & Cooking
    (["pizza", "burger", "sushi", "taco", "sandwich", "hot dog", "pasta",
      "noodle", "rice", "soup", "steak", "barbecue", "salad", "bread",
      "croissant", "pretzel", "bagel", "burrito", "dumpling",
      "cheeseburger", "french fries", "guacamole", "meatloaf",
      "potpie", "carbonara"], "Food", "food"),
    (["cake", "ice cream", "chocolate", "cookie", "donut", "cupcake",
      "waffle", "pancake", "trifle", "custard"], "Food", "desserts"),
    (["banana", "apple", "orange", "strawberry", "pineapple", "lemon",
      "grape", "watermelon", "mango", "peach", "cherry", "pomegranate",
      "fig", "coconut", "avocado", "kiwi"], "Food", "fruits"),
    (["espresso", "coffee", "latte", "cappuccino", "cup", "mug",
      "wine", "beer", "cocktail", "juice", "smoothie", "tea",
      "eggnog"], "Food", "drinks"),
    # Sports & Fitness
    (["soccer", "football", "basketball", "tennis", "baseball", "volleyball",
      "golf", "rugby", "cricket ball", "ping-pong", "badminton",
      "hockey"], "Sports", "ball-sports"),
    (["swimming", "surfing", "diving", "kayak", "canoe",
      "paddleboard", "snorkel", "scuba"], "Sports", "water-sports"),
    (["bicycle", "mountain bike", "cycling", "unicycle"], "Sports", "cycling"),
    (["ski", "snowboard", "sled", "bobsled"], "Sports", "winter-sports"),
    (["gym", "dumbbell", "barbell", "weight", "treadmill"], "Sports", "fitness"),
    (["boxing", "wrestling", "martial", "karate", "judo",
      "taekwondo", "fencing"], "Sports", "combat-sports"),
    # Vehicles & Travel
    (["car", "sports car", "convertible", "limousine", "taxi", "cab",
      "minivan", "SUV", "jeep", "pickup", "race car", "go-kart",
      "ambulance", "fire engine", "police", "garbage truck",
      "tow truck", "trailer truck", "moving van"], "Vehicles", "cars"),
    (["motorcycle", "scooter", "moped", "motor scooter"], "Vehicles", "motorcycles"),
    (["airplane", "airliner", "aircraft", "jet", "biplane",
      "helicopter", "airship"], "Travel", "aviation"),
    (["boat", "ship", "yacht", "sailboat", "canoe", "gondola",
      "submarine", "ferry", "catamaran", "speedboat"], "Travel", "boats"),
    (["train", "locomotive", "bullet train", "freight car",
      "passenger car", "trolleybus", "streetcar"], "Travel", "trains"),
    # Technology & Gadgets
    (["laptop", "notebook", "computer", "desktop computer", "screen",
      "monitor", "keyboard", "mouse"], "Technology", "computers"),
    (["cellphone", "cell phone", "smartphone", "iPhone",
      "mobile phone", "dial telephone"], "Technology", "phones"),
    (["camera", "reflex camera", "Polaroid", "projector",
      "lens", "tripod"], "Technology", "photography"),
    (["television", "TV", "remote control", "joystick",
      "game controller"], "Technology", "entertainment-tech"),
    (["robot", "drone", "space shuttle", "satellite",
      "solar dish", "radar"], "Technology", "robotics"),
    # Music & Performance
    (["guitar", "electric guitar", "acoustic guitar", "bass guitar",
      "banjo", "ukulele", "mandolin", "harp", "sitar", "lyre",
      "piano", "grand piano", "organ", "accordion", "harmonica",
      "violin", "cello", "flute", "oboe", "saxophone", "trumpet",
      "trombone", "French horn", "tuba", "clarinet", "drum",
      "drumstick", "maraca", "cymbal", "gong", "steel drum",
      "marimba", "xylophone"], "Music", "instruments"),
    (["stage", "microphone", "spotlight", "concert"], "Music", "performance"),
    # Fashion & Beauty
    (["dress", "gown", "kimono", "bikini", "swimsuit", "jersey",
      "suit", "tuxedo", "lab coat", "trench coat", "poncho",
      "cloak", "fur coat"], "Fashion", "clothing"),
    (["sneaker", "running shoe", "boot", "sandal", "clog",
      "loafer", "high heel", "flip-flop"], "Fashion", "shoes"),
    (["sunglasses", "sunglass", "hat", "cowboy hat", "sombrero",
      "beret", "bonnet", "crown", "tiara", "wig", "necklace",
      "bow tie", "necktie", "scarf"], "Fashion", "accessories"),
    (["lipstick", "lotion", "perfume", "hair spray",
      "face powder", "makeup"], "Fashion", "beauty"),
    # Architecture & Urban
    (["church", "mosque", "monastery", "palace", "castle", "fortress",
      "lighthouse", "dome", "bell tower", "triumphal arch",
      "temple"], "Architecture", "historical"),
    (["skyscraper", "building", "tower", "bridge", "suspension bridge",
      "viaduct", "steel arch bridge"], "Architecture", "modern"),
    (["barn", "greenhouse", "cottage", "mobile home", "mansion",
      "estate", "log cabin", "chalet"], "Architecture", "houses"),
    # Education & Office
    (["library", "bookshop", "book", "notebook", "binder",
      "pencil", "pen", "ruler", "eraser", "backpack",
      "desk", "filing cabinet"], "Education", "study"),
    (["whiteboard", "blackboard", "chalk", "classroom",
      "lecture hall", "podium", "easel"], "Education", "classroom"),
    # Gaming & Entertainment
    (["video game", "arcade", "pinball", "slot", "console",
      "controller", "joystick", "chess", "crossword",
      "jigsaw", "Rubik"], "Gaming", "gaming"),
    # Art & Creativity
    (["paintbrush", "paint", "palette", "canvas", "easel",
      "crayon", "quill", "fountain pen"], "Art", "art-supplies"),
    (["sculpture", "statue", "pottery", "vase", "ceramic",
      "mosaic", "stained glass"], "Art", "sculpture"),
]


# ──────────────────────────────────────────────────────────────────────
# 3. Analysis Helper Functions
# ──────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def deduplicate_embeddings(embeddings: list[np.ndarray], threshold: float = INTRA_VIDEO_SIMILARITY_THRESHOLD) -> list[np.ndarray]:
    """Merge same-person embeddings within a single video."""
    if not embeddings:
        return []
    clusters: list[list[np.ndarray]] = []
    for emb in embeddings:
        matched = False
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            if cosine_similarity(emb, centroid) >= threshold:
                cluster.append(emb)
                matched = True
                break
        if not matched:
            clusters.append([emb])
    unique_embeddings = []
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        unique_embeddings.append(centroid)
    return unique_embeddings


def classify_frame_scene(frame: np.ndarray) -> list[tuple[str, float]]:
    """Classify a single frame using MobileNetV3 against ImageNet. Returns top-5."""
    labels = get_imagenet_labels()
    if not labels:
        return []
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = scene_transform(rgb_frame).unsqueeze(0)
        with torch.no_grad():
            output = scene_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            class_name = labels[idx.item()]
            results.append((class_name, prob.item()))
        return results
    except Exception as e:
        print(f"Scene classification error: {e}")
        return []


def extract_text_with_ocr(frame: np.ndarray) -> str:
    """Extract text from the frame using EasyOCR."""
    try:
        results = ocr_reader.readtext(frame, detail=0, paragraph=True)
        return " ".join(results)
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def map_predictions_to_tags(all_predictions: list[tuple[str, float]]) -> tuple[str, list[str]]:
    """Map accumulated ImageNet predictions to content categories and tags."""
    if not all_predictions:
        return "Entertainment", ["general"]
    category_scores: dict[str, float] = {}
    tag_scores: dict[str, float] = {}
    for class_name, confidence in all_predictions:
        class_lower = class_name.lower()
        for keywords, category, tag in SCENE_CATEGORY_MAP:
            for keyword in keywords:
                if keyword.lower() in class_lower:
                    category_scores[category] = category_scores.get(category, 0.0) + confidence
                    tag_scores[tag] = tag_scores.get(tag, 0.0) + confidence
                    break
    if not category_scores:
        return "Entertainment", ["general"]
    best_category = max(category_scores, key=category_scores.get)
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, score in sorted_tags[:3] if score > 0.05]
    if not top_tags:
        top_tags = ["general"]
    return best_category, top_tags


# ──────────────────────────────────────────────────────────────────────
# 4. NEW: Perceptual Hash (pHash) for Duplicate Video Detection
# ──────────────────────────────────────────────────────────────────────

def compute_phash(frame: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash for a frame using DCT.
    Produces a 64-bit hex string that is similar for visually similar images.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        # Resize to 32x32 for DCT
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        resized = np.float32(resized)
        # Apply 2D DCT
        dct = cv2.dct(resized)
        # Take the top-left 8x8 (low frequency components — the "gist" of the image)
        dct_low = dct[:hash_size, :hash_size]
        # Compute median (excluding DC component at [0,0])
        median_val = np.median(dct_low)
        # Build binary hash: 1 if above median, 0 if below
        hash_bits = (dct_low > median_val).flatten()
        # Convert to hex string
        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)
        return format(hash_int, f'0{hash_size * hash_size // 4}x')
    except Exception as e:
        print(f"pHash error: {e}")
        return ""


def compute_video_fingerprint(frame_hashes: list[str]) -> str:
    """
    Combine per-frame perceptual hashes into a single video fingerprint.
    Uses the most common hash (mode) as the representative.
    Then appends a secondary hash from the temporal sequence for uniqueness.
    """
    if not frame_hashes:
        return ""
    # Find the most common frame hash (the dominant visual)
    from collections import Counter
    counter = Counter(frame_hashes)
    dominant_hash = counter.most_common(1)[0][0]
    # Create a temporal fingerprint from the sequence of hashes
    import hashlib
    temporal = hashlib.md5("_".join(frame_hashes).encode()).hexdigest()[:8]
    return f"{dominant_hash}_{temporal}"


# ──────────────────────────────────────────────────────────────────────
# 5. NEW: Thumbnail Selection (best frame picker)
# ──────────────────────────────────────────────────────────────────────

def score_frame_for_thumbnail(frame: np.ndarray, gray: np.ndarray, has_face: bool) -> float:
    """
    Score a frame's suitability as a thumbnail based on:
    - Sharpness (Laplacian variance) — crisp frames beat blurry ones
    - Color vibrancy (saturation in HSV) — vibrant frames are more eye-catching
    - Face presence — frames with faces make better thumbnails
    - Brightness — not too dark, not too bright
    """
    score = 0.0

    # Sharpness via Laplacian variance (higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize, cap at 1.0
    score += sharpness_score * 30.0  # Weight: 30

    # Color vibrancy via mean saturation in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
    score += avg_saturation * 20.0  # Weight: 20

    # Brightness — penalize too dark (<40) or too bright (>220)
    avg_brightness = np.mean(gray)
    if avg_brightness < 40:
        score -= 15.0
    elif avg_brightness > 220:
        score -= 10.0
    else:
        # Ideal range: 80-180
        brightness_quality = 1.0 - abs(avg_brightness - 130) / 130.0
        score += brightness_quality * 15.0  # Weight: 15

    # Face bonus
    if has_face:
        score += 25.0  # Weight: 25

    return score


# ──────────────────────────────────────────────────────────────────────
# 6. NEW: Video Quality Score
# ──────────────────────────────────────────────────────────────────────

def compute_quality_score(
    width: int, height: int,
    sharpness_values: list[float],
    brightness_values: list[float],
    noise_values: list[float]
) -> float:
    """
    Compute an overall video quality score from 0.0 to 1.0 based on:
    - Resolution (720p+ is good, 1080p+ is great)
    - Sharpness (average Laplacian variance across frames)
    - Brightness consistency (not too dark/bright)
    - Noise level (lower is better)
    """
    # Resolution score: 0.0-0.25
    pixels = width * height
    if pixels >= 1920 * 1080:    # 1080p+
        resolution_score = 0.25
    elif pixels >= 1280 * 720:   # 720p
        resolution_score = 0.20
    elif pixels >= 854 * 480:    # 480p
        resolution_score = 0.12
    else:
        resolution_score = 0.05

    # Sharpness score: 0.0-0.30
    avg_sharpness = np.mean(sharpness_values) if sharpness_values else 0
    sharpness_score = min(avg_sharpness / 300.0, 1.0) * 0.30

    # Brightness score: 0.0-0.25 (penalize extremes)
    if brightness_values:
        avg_brightness = np.mean(brightness_values)
        # Ideal brightness is around 100-160
        if 80 <= avg_brightness <= 180:
            brightness_score = 0.25
        elif 40 <= avg_brightness <= 220:
            brightness_score = 0.15
        else:
            brightness_score = 0.05
    else:
        brightness_score = 0.15

    # Noise score: 0.0-0.20 (lower noise = higher score)
    if noise_values:
        avg_noise = np.mean(noise_values)
        noise_score = max(0.0, (1.0 - avg_noise / 50.0)) * 0.20
    else:
        noise_score = 0.10

    total = resolution_score + sharpness_score + brightness_score + noise_score
    return round(min(total, 1.0), 3)


def estimate_noise(gray_frame: np.ndarray) -> float:
    """Estimate noise level using Laplacian-based noise estimation."""
    try:
        # Noise estimation via median absolute deviation of Laplacian
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) * 1.4826
        return float(sigma)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────
# 7. NEW: Color/Mood Analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_color_mood(frames_hsv_stats: list[tuple[float, float, float]]) -> tuple[str, list[str]]:
    """
    Analyze color mood from collected HSV statistics across frames.
    
    Args:
        frames_hsv_stats: list of (avg_hue, avg_saturation, avg_value) tuples
    
    Returns:
        (mood_label, dominant_color_hex_list)
    """
    if not frames_hsv_stats:
        return "neutral", []

    avg_h = np.mean([s[0] for s in frames_hsv_stats])
    avg_s = np.mean([s[1] for s in frames_hsv_stats])
    avg_v = np.mean([s[2] for s in frames_hsv_stats])

    # Classify mood based on HSV ranges
    if avg_v < 80:
        mood = "dark"
    elif avg_s < 50 and avg_v > 180:
        mood = "pastel"
    elif avg_s > 150 and avg_v > 120:
        mood = "vibrant"
    elif avg_h < 30 or avg_h > 160:  # Reds, oranges, yellows
        mood = "warm" if avg_s > 80 else "neutral"
    elif 90 < avg_h < 150:  # Blues, cyans
        mood = "cool" if avg_s > 80 else "neutral"
    else:
        mood = "neutral"

    return mood, []


def extract_dominant_colors(frame: np.ndarray, k: int = 3) -> list[str]:
    """
    Extract top-k dominant colors from a frame using K-means clustering.
    Returns list of hex color strings.
    """
    try:
        # Resize for speed
        small = cv2.resize(frame, (64, 64))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        # Sort by frequency (most dominant first)
        _, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)

        hex_colors = []
        for idx in sorted_indices:
            b, g, r = centers[idx].astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hex_colors.append(hex_color)

        return hex_colors
    except Exception as e:
        print(f"Dominant color extraction error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────
# 8. NEW: Motion Intensity Analysis
# ──────────────────────────────────────────────────────────────────────

def classify_motion_intensity(motion_scores: list[float]) -> str:
    """
    Classify overall motion intensity from per-frame motion scores.
    Returns: "static", "low", "medium", "high"
    """
    if not motion_scores:
        return "static"

    avg_motion = np.mean(motion_scores)
    max_motion = np.max(motion_scores)

    if avg_motion < 0.02 and max_motion < 0.05:
        return "static"
    elif avg_motion < 0.08:
        return "low"
    elif avg_motion < 0.20:
        return "medium"
    else:
        return "high"


# ──────────────────────────────────────────────────────────────────────
# Main Analysis Function
# ──────────────────────────────────────────────────────────────────────

def analyze_video_frames(video_url: str):
    """
    Full video analysis pipeline:
    - InsightFace: face detection + embedding
    - MobileNetV3: scene/content classification
    - Text detection: caption/meme detection
    - pHash: visual fingerprinting for duplicate detection
    - Thumbnail selection: best frame scoring
    - Quality scoring: resolution, sharpness, brightness, noise
    - Color/mood analysis: HSV-based mood + dominant colors
    - Motion intensity: optical flow-based motion analysis
    - Cut detection: edit pacing
    
    Returns:
        category, tags, is_nsfw, face_embeddings, extra_analysis
    """

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print(f"Error opening video URL: {video_url}")
        return "Entertainment", ["error"], False, [], {}

    # Video resolution
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or video_fps != video_fps:
        video_fps = 30.0

    frame_count = 0
    faces_detected = 0
    cuts_detected = 0
    prev_gray = None

    # Collection arrays
    all_face_embeddings: list[np.ndarray] = []
    all_scene_predictions: list[tuple[str, float]] = []
    all_yolo_tags: set[str] = set()
    ocr_texts: list[str] = []
    max_nsfw_score = 0.0
    
    pose_wrist_y: list[float] = []
    pose_ankle_y: list[float] = []
    
    frame_hashes: list[str] = []
    sharpness_values: list[float] = []
    brightness_values: list[float] = []
    noise_values: list[float] = []
    hsv_stats: list[tuple[float, float, float]] = []
    motion_scores: list[float] = []
    dominant_colors_sample: list[str] = []

    # Thumbnail tracking
    best_thumbnail_score = -999.0
    best_thumbnail_timestamp_ms = 0

    frame_interval = max(1, int(video_fps / 3))
    scene_classify_interval = 5

    while cap.isOpened() and frame_count < 45:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + frame_interval)

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # ── Face Detection ─────────────────────────────────────────
        frame_has_face = False
        try:
            faces = face_app.get(frame)
            if faces:
                faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:MAX_FACES_PER_FRAME]
                faces_detected += len(faces)
                frame_has_face = True
                for face in faces:
                    if face.embedding is not None:
                        all_face_embeddings.append(face.embedding)
        except Exception as e:
            print(f"Face detection error on frame {frame_count}: {e}")

        # ── Scene Classification & YOLO (every 5th frame) ──────────────
        if frame_count % scene_classify_interval == 0:
            predictions = classify_frame_scene(frame)
            all_scene_predictions.extend(predictions)
            
            try:
                yolo_results = yolo_model(frame, verbose=False)
                for r in yolo_results:
                    for cls_id in r.boxes.cls:
                        class_name = yolo_model.names[int(cls_id)]
                        all_yolo_tags.add(class_name)
            except Exception as e:
                print(f"YOLO error: {e}")

            try:
                pose_results = pose_model(frame, verbose=False)
                for r in pose_results:
                    if r.keypoints is not None and r.keypoints.xy is not None:
                        for kpts in r.keypoints.xy:
                            if len(kpts) >= 17:
                                lw_y = float(kpts[9][1])
                                rw_y = float(kpts[10][1])
                                la_y = float(kpts[15][1])
                                ra_y = float(kpts[16][1])
                                
                                # 0 means not detected
                                if lw_y > 0 and rw_y > 0:
                                    pose_wrist_y.append((lw_y + rw_y) / 2)
                                if la_y > 0 and ra_y > 0:
                                    pose_ankle_y.append((la_y + ra_y) / 2)
            except Exception as e:
                print(f"Pose error: {e}")

        # ── NSFW Detection (every 10th frame) ──────────────────────
        if frame_count % 10 == 0:
            try:
                # NudeDetector expects image path or BGR frame. 
                preds = nude_detector.detect(frame)
                for p in preds:
                    if p['score'] > max_nsfw_score:
                        max_nsfw_score = float(p['score'])
            except Exception as e:
                pass

        # ── OCR (frame 15 and 30) ──────────────────────────────────
        if frame_count in (15, 30):
            text = extract_text_with_ocr(frame)
            if text:
                ocr_texts.append(text)

        # ── Perceptual Hash ────────────────────────────────────────
        phash = compute_phash(frame)
        if phash:
            frame_hashes.append(phash)

        # ── Quality Metrics ────────────────────────────────────────
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_values.append(laplacian_var)
        brightness_values.append(float(np.mean(gray)))
        noise_values.append(estimate_noise(gray))

        # ── Color/Mood (HSV stats) ─────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_h = float(np.mean(hsv[:, :, 0]))
        avg_s = float(np.mean(hsv[:, :, 1]))
        avg_v = float(np.mean(hsv[:, :, 2]))
        hsv_stats.append((avg_h, avg_s, avg_v))

        # Extract dominant colors from the middle frame
        if frame_count == 15 or (frame_count == 0 and not dominant_colors_sample):
            dominant_colors_sample = extract_dominant_colors(frame, k=3)

        # ── Thumbnail Scoring ──────────────────────────────────────
        thumb_score = score_frame_for_thumbnail(frame, gray, frame_has_face)
        if thumb_score > best_thumbnail_score:
            best_thumbnail_score = thumb_score
            best_thumbnail_timestamp_ms = current_timestamp_ms

        # ── Motion Analysis ────────────────────────────────────────
        if prev_gray is not None:
            curr_small = cv2.resize(gray, (64, 64))
            prev_small = cv2.resize(prev_gray, (64, 64))

            diff = cv2.absdiff(curr_small, prev_small)
            non_zero_count = np.count_nonzero(diff > 30)
            total_pixels = 64 * 64

            # Motion score: fraction of pixels that changed significantly
            motion_score = non_zero_count / total_pixels
            motion_scores.append(motion_score)

            # Cut detection (>30% change)
            if motion_score > 0.3:
                cuts_detected += 1

        prev_gray = gray
        frame_count += 1

    cap.release()

    # ══════════════════════════════════════════════════════════════════
    # Build results
    # ══════════════════════════════════════════════════════════════════

    tags = []
    category = "Entertainment"

    has_faces = faces_detected > (frame_count * 0.2) if frame_count > 0 else False
    
    # Compile OCR text
    final_ocr_text = " ".join(ocr_texts).strip()
    has_text = len(final_ocr_text) > 5

    # Scene classification (Tags ONLY)
    _, scene_tags = map_predictions_to_tags(all_scene_predictions)
        
    # Combine scene tags and YOLO tags
    combined_tags = set(scene_tags)
    for t in all_yolo_tags:
        combined_tags.add(t.lower())
    
    for tag in combined_tags:
        if tag not in tags and len(tags) < 15:
            tags.append(tag)

    # Text tags
    if has_text:
        tags.append("text-overlay")

    # Edit style tags
    if cuts_detected > 3:
        tags.append("fast-paced-edit")
    elif cuts_detected == 0:
        tags.append("continuous-shot")

    # Motion intensity tag
    motion_intensity = classify_motion_intensity(motion_scores)
    if motion_intensity in ("high", "medium"):
        tags.append(f"{motion_intensity}-motion")

    # Calculate Dance Confidence Score based on Pose Variance
    dance_confidence = 0.0
    if len(pose_wrist_y) > 2:
        wrist_var = float(np.var(pose_wrist_y))
        
        # Normalize relative to video height
        wrist_var_norm = wrist_var / max(1.0, float(video_height * video_height))
        
        # If variance is erratic, it's a dance
        if wrist_var_norm > 0.002:
            dance_confidence = min(1.0, wrist_var_norm * 100)
            
    is_dancing_pose = dance_confidence > 0.4

    # ACTION-BASED CATEGORIZATION (Visual Base)
    if is_dancing_pose:
        category = "Entertainment"
        if "dancing" not in tags: tags.append("dancing")
        if "performance" not in tags: tags.append("performance")
    elif len(final_ocr_text) > 30 and motion_intensity == "low":
        category = "Education"
    elif has_faces and motion_intensity == "high":
        category = "Entertainment"
    elif has_faces and motion_intensity in ("low", "medium"):
        category = "Lifestyle"
        if "vlog" not in tags: tags.append("vlog")
    elif motion_intensity == "high" and cuts_detected > 2:
        category = "Action"
    else:
        category = "Entertainment"

    if has_faces and "face-centric" not in tags:
        tags.append("face-centric")

    # NSFW
    is_nsfw = max_nsfw_score > 0.6

    if len(tags) < 2:
        tags.append("trending")

    # Face embeddings
    unique_embeddings = deduplicate_embeddings(all_face_embeddings)
    face_embedding_lists = [emb.tolist() for emb in unique_embeddings]

    # Visual fingerprint
    visual_fingerprint = compute_video_fingerprint(frame_hashes)

    # Quality score
    quality_score = compute_quality_score(video_width, video_height, sharpness_values, brightness_values, noise_values)

    # Color mood
    color_mood, _ = analyze_color_mood(hsv_stats)
    
    # Engagement Score prediction
    engagement_score = 0.05
    if has_faces: engagement_score += 0.15
    engagement_score += min(quality_score * 0.15, 0.15)
    if motion_intensity in ("high", "medium"): engagement_score += 0.10
    if cuts_detected > 2: engagement_score += 0.10
    if color_mood == "vibrant": engagement_score += 0.05
    if has_text: engagement_score += 0.05
    if len(all_yolo_tags) > 2: engagement_score += 0.10
    if (video_width * video_height) >= (1280 * 720): engagement_score += 0.05
    engagement_score = round(min(engagement_score, 1.0), 3)

    # Extra analysis payload
    extra_analysis = {
        "visualFingerprint": visual_fingerprint,
        "thumbnailTimestampMs": best_thumbnail_timestamp_ms,
        "qualityScore": quality_score,
        "colorMood": color_mood,
        "dominantColors": dominant_colors_sample,
        "motionIntensity": motion_intensity,
        "nsfwScore": round(max_nsfw_score, 3),
        "ocrText": final_ocr_text,
        "engagementScore": engagement_score,
        "danceConfidence": round(dance_confidence, 3),
    }

    print(f"CV Stats - Frames: {frame_count}, Faces: {faces_detected}, "
          f"Cuts: {cuts_detected}, Unique faces: {len(face_embedding_lists)}, "
          f"Category: {category}, Tags: {tags}, "
          f"Quality: {quality_score}, Mood: {color_mood}, "
          f"Motion: {motion_intensity}, NSFW Score: {max_nsfw_score:.2f}, "
          f"OCR Length: {len(final_ocr_text)}, Engagement: {engagement_score:.2f}")

    return category, tags, is_nsfw, face_embedding_lists, extra_analysis
