================================================================
  ONION DISEASE DATASET
  AI Agricultural Triage System — Layer 3
  Crop: Allium cepa (Onion)
================================================================

FOLDER STRUCTURE
----------------
data/
└── onion/
    ├── images/
    │   ├── 1_purple_blotch/        ← PUT IMAGES HERE
    │   ├── 2_downy_mildew/         ← PUT IMAGES HERE
    │   ├── 3_stemphylium_blight/   ← PUT IMAGES HERE
    │   ├── 4_botrytis_leaf_blight/ ← PUT IMAGES HERE
    │   └── 5_healthy/              ← PUT IMAGES HERE
    │
    └── disease_info/
        ├── 1_purple_blotch.txt     ← Full disease description + chemicals
        ├── 2_downy_mildew.txt      ← Full disease description + chemicals
        ├── 3_stemphylium_blight.txt← Full disease description + chemicals
        ├── 4_botrytis_leaf_blight.txt
        └── 5_healthy.txt

================================================================
  STEP 1 — GET IMAGES (do this first)
================================================================

Download the Kaggle dataset:
  URL: https://www.kaggle.com/datasets/tejasbargujepatil/onion-diseases

  Using Kaggle CLI:
    kaggle datasets download tejasbargujepatil/onion-diseases
    unzip onion-diseases.zip

  Then copy each class folder into the matching images folder:
    purple_blotch/   → data/onion/images/1_purple_blotch/
    downy_mildew/    → data/onion/images/2_downy_mildew/
    stemphylium/     → data/onion/images/3_stemphylium_blight/
    botrytis/        → data/onion/images/4_botrytis_leaf_blight/
    healthy/         → data/onion/images/5_healthy/

================================================================
  STEP 2 — WHAT IS MISSING AFTER KAGGLE
================================================================

After Kaggle download you will have roughly 200–400 images per
class. You need 800–1,000 per class for solid training.
Here is what to do for each missing class:

CLASS 1 — Purple Blotch (~600 still needed)
  → bugwood.org — search: "Alternaria porri"
  → ipmimages.org — search: "purple blotch onion"

CLASS 2 — Downy Mildew (~600 still needed)
  → bugwood.org — search: "Peronospora destructor onion"
  → mtvernon.wsu.edu/path_team/diseasegallery.htm

CLASS 3 — Stemphylium Blight (~700 still needed) ← HARDEST
  → vegpath.plantpath.wisc.edu — onion stemphylium page
  → bugwood.org — search: "Stemphylium vesicarium"
  → If still short: use augmentation (see below)

CLASS 4 — Botrytis Leaf Blight (~600 still needed)
  → ontario.ca — identification diseases onions page
  → bugwood.org — search: "Botrytis squamosa onion leaf"
  ⚠ Only leaf blight images, NOT neck rot (storage disease)

CLASS 5 — Healthy (~500 still needed)
  → Take your own field photos (fastest option)
  → openimages — search: "onion"
  → inaturalist.org — search: Allium cepa

================================================================
  AUGMENTATION (for Stemphylium and any short class)
================================================================

Safe augmentations for plant disease images:
  ✓ Horizontal flip
  ✓ Vertical flip
  ✓ Random 90° rotation
  ✓ Color jitter — brightness ±30%, contrast ±30%, saturation ±30%
  ✓ Random crop (resize to 256px, crop to 224px)
  ✓ Mild Gaussian blur (p=0.2)

Do NOT use:
  ✗ Strong hue shifts — the purple/brown color IS the diagnosis
  ✗ Grayscale — color is a diagnostic feature
  ✗ Large CutOut — may erase the lesion itself

Python (albumentations):
  import albumentations as A
  transform = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
      A.GaussianBlur(blur_limit=3, p=0.2),
  ])

================================================================
  IMAGE REQUIREMENTS
================================================================

  Format     : JPG or PNG
  Min size   : 224 x 224 pixels
  Color      : RGB (no grayscale)
  Target     : 800–1,000 images per class
  Naming     : pb_001.jpg / dm_001.jpg / st_001.jpg /
               bt_001.jpg / hl_001.jpg

================================================================
  PyTorch USAGE
================================================================

  from torchvision.datasets import ImageFolder
  dataset = ImageFolder(
      root='data/onion/images',
      transform=transform
  )
  # Auto-detected classes (sorted by folder name):
  # 0 = 1_purple_blotch
  # 1 = 2_downy_mildew
  # 2 = 3_stemphylium_blight
  # 3 = 4_botrytis_leaf_blight
  # 4 = 5_healthy

================================================================
  DISEASE INFO FILES (disease_info folder)
================================================================

Each .txt file in disease_info/ contains:
  - Scientific name and pathogen type
  - Full visual symptom description
  - Favorable outbreak conditions
  - Agronomic prevention methods
  - Chemical treatment products with:
      * Product name (Hebrew / trade name)
      * Active ingredient
      * FRAC resistance group code
      * Dose per dunam (1,000 m²)
      * Mode of action
      * Rotation strategy

Sources:
  Madrich Machalot Gidulim HE (full disease guide)
  Hafes Rases / Agrica crop guide (registered Israeli products)
  Israeli Ministry of Agriculture pesticide database

================================================================
  CONFUSION PAIRS — labeling and model risks
================================================================

  Stemphylium ↔ Purple Blotch
    Key: Purple Blotch HAS a purple halo. Stemphylium does NOT.

  Stemphylium ↔ Botrytis (tip)
    Key: Botrytis has white spots. Stemphylium has brown-black.

  Downy Mildew ↔ early Healthy
    Key: Downy Mildew has grey coating in morning. Healthy does not.

  Botrytis Leaf Blight ↔ Botrytis Neck Rot
    Key: Leaf Blight = white spots on leaves.
         Neck Rot = soft rot on stored bulb. Different disease.

================================================================
