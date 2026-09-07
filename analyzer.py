"""ROK Hospital OCR analyzer.

Cleaned runtime analyzer used by the Discord bot.
It isolates the hospital window first, reads troop lines, classifies T4/T5
from unit names, and reads visible heal resources from their icons.
Supports 1 or 2 screenshots and PC/phone layouts.
"""

import os
import re
import sys
import shutil
import unicodedata
from collections import Counter

import cv2
import numpy as np
import pytesseract


# =========================================================
# TESSERACT CONFIGURATION
# =========================================================

_tesseract_env = os.getenv("TESSERACT_CMD")

if _tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_env
else:
    _tesseract_path = shutil.which("tesseract")
    if _tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path
    else:
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )


T4_UNITS = [

    # =====================================================
    # ANGLAIS
    # =====================================================

    "Long Swordsman",
    "Legionary",
    "Throwing Axeman",
    "Samurai",
    "Berserker",
    "Argyraspides",

    "Crossbowman",
    "Longbowman",
    "Chu-Ko-Nu",
    "Hwarang",
    "Janissary",

    "Maryannu",
    "Spear-Thrower",

    "Knight",
    "Teutonic Knight",
    "Conquistador",
    "Mamluk",
    "Cataphract",

    "Ballista",

    # =====================================================
    # FRANCAIS
    # =====================================================

    "Bretteur",
    "Légionnaire",
    "Lanceur de haches",
    "Samouraï",
    "Berserker",
    "Argyraspide",

    "Arbalétrier",
    "Archer à arc long",
    "Chu-ko-nu",
    "Hwarang",
    "Janissaire",

    "Maryannu",
    "Lanceur de sagaies",

    "Chevalier",
    "Chevalier Teutonique",
    "Conquistador",
    "Mamelouk",
    "Cataphractaire",

    "Baliste",
]

T5_UNITS = [

    # =====================================================
    # ANGLAIS
    # =====================================================

    "Royal Guard",
    "Elite Legionary",
    "Elite Throwing Axeman",
    "Elite Samurai",

    "Royal Crossbowman",
    "Elite Longbowman",
    "Elite Chu-Ko-Nu",
    "Elite Hwarang",
    "Elite Janissary",

    "Elite Maryannu",
    "Elite Spear-Thrower",

    "Royal Knight",
    "Elite Teutonic Knight",
    "Elite Conquistador",
    "Elite Mamluk",
    "Elite Cataphract",

    "Trebuchet",

    # =====================================================
    # FRANCAIS
    # =====================================================

    "Garde Royale",
    "Légionnaire d'élite",
    "Lanceur de haches d'élite",
    "Samouraï d'élite",

    "Arbalétrier royal",
    "Archer à arc long d'élite",
    "Chu-ko-nu d'élite",
    "Hwarang d'élite",
    "Janissaire d'élite",

    "Maryannu d'élite",
    "Lanceur de sagaies d'élite",

    "Chevalier royal",
    "Chevalier teutonique d'élite",
    "Conquistador d'élite",
    "Mamelouk d'élite",
    "Cataphractaire d'élite",

    "Trébuchet",
]

UNIT_TIERS = {}

UNIT_NAMES_SORTED = []


# =========================================================
# NORMALISER TEXTE
# =========================================================

def normaliser_texte(texte):

    texte = unicodedata.normalize(
        "NFKD",
        texte
    )

    texte = (
        texte
        .encode(
            "ascii",
            "ignore"
        )
        .decode(
            "ascii"
        )
    )

    texte = texte.lower()

    texte = re.sub(
        r"[^a-z0-9]+",
        " ",
        texte
    )

    texte = re.sub(
        r"\s+",
        " ",
        texte
    )

    return texte.strip()


# Construire la table T4/T5 après la définition de normaliser_texte().
for nom in T4_UNITS:
    UNIT_TIERS[
        normaliser_texte(nom)
    ] = "T4"

for nom in T5_UNITS:
    UNIT_TIERS[
        normaliser_texte(nom)
    ] = "T5"

UNIT_NAMES_SORTED = sorted(
    UNIT_TIERS.keys(),
    key=len,
    reverse=True
)


# =========================================================
# PREPARER CROP
# =========================================================

def preparer_crop(
    image,
    x1,
    y1,
    x2,
    y2,
    scale=6
):

    h, w = image.shape[:2]

    xx1 = max(
        0,
        int(
            w * x1
        )
    )

    xx2 = min(
        w,
        int(
            w * x2
        )
    )

    yy1 = max(
        0,
        int(
            h * y1
        )
    )

    yy2 = min(
        h,
        int(
            h * y2
        )
    )

    if (
        xx2 <= xx1
        or
        yy2 <= yy1
    ):

        return None

    crop = image[
        yy1:yy2,
        xx1:xx2
    ]

    if crop.size == 0:

        return None

    return cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


# =========================================================
# DETECTER LIGNES
# =========================================================

def detecter_lignes(
    image
):

    h, w = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Vert des barres de vie
    masque_vert = cv2.inRange(
        hsv,
        np.array(
            [40, 100, 70]
        ),
        np.array(
            [90, 255, 255]
        )
    )

    # Zone horizontale des barres
    x1 = int(
        w * 0.48
    )

    x2 = int(
        w * 0.92
    )

    zone = masque_vert[
        :,
        x1:x2
    ]

    projection = np.sum(
        zone > 0,
        axis=1
    )

    seuil = max(
        40,
        int(
            (x2 - x1)
            * 0.15
        )
    )

    lignes = []

    en_cours = False
    debut = None

    for y, valeur in enumerate(
        projection
    ):

        if valeur > seuil:

            if not en_cours:

                en_cours = True
                debut = y

        else:

            if en_cours:

                fin = y

                if (
                    fin - debut
                    >= 3
                ):

                    lignes.append(
                        (
                            debut
                            +
                            fin
                        )
                        //
                        2
                    )

                en_cours = False

    if en_cours:

        fin = len(
            projection
        )

        if (
            fin - debut
            >= 3
        ):

            lignes.append(
                (
                    debut
                    +
                    fin
                )
                //
                2
            )

    # Fusion des détections proches
    resultat = []

    for y in lignes:

        if not resultat:

            resultat.append(
                y
            )

        elif (
            abs(
                y
                -
                resultat[-1]
            )
            > 20
        ):

            resultat.append(
                y
            )

        else:

            resultat[-1] = (
                resultat[-1]
                +
                y
            ) // 2

    return resultat


# =========================================================
# OCR PANNEAU TROUPES
# =========================================================

def ocr_panneau_troupes(
    image
):
    """
    OCR des noms et quantités des troupes, robuste aux petites captures.
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.30)
    x2 = int(w * 0.985)
    y1 = int(h * 0.07)
    y2 = int(h * 0.78)

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return []

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    variantes = [
        crop,
        gray,
        clahe.apply(gray)
    ]

    for block in (
        21,
        31
    ):

        b = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            6
        )

        variantes.append(
            b
        )

        variantes.append(
            cv2.bitwise_not(b)
        )

    mots = []

    for variante in variantes:

        for psm in (
            6,
            11
        ):

            data = pytesseract.image_to_data(
                variante,
                config=f"--psm {psm}",
                output_type=pytesseract.Output.DICT
            )

            for i, brut in enumerate(
                data["text"]
            ):

                brut = brut.strip()

                if not brut:
                    continue

                try:
                    confiance = float(
                        data["conf"][i]
                    )
                except (
                    ValueError,
                    TypeError
                ):
                    confiance = 0.0

                largeur = float(
                    data["width"][i]
                )

                hauteur = float(
                    data["height"][i]
                )

                if (
                    largeur <= 0
                    or
                    hauteur <= 0
                ):
                    continue

                mots.append(
                    {
                        "texte":
                            brut,

                        "x":
                            float(
                                data["left"][i]
                                +
                                x1
                            ),

                        "y":
                            float(
                                data["top"][i]
                                +
                                hauteur / 2
                                +
                                y1
                            ),

                        "largeur":
                            largeur,

                        "hauteur":
                            hauteur,

                        "confiance":
                            confiance
                    }
                )

    resultat = []

    for mot in mots:

        ancien = None

        for candidat in resultat:

            if (
                normaliser_texte(
                    mot["texte"]
                )
                ==
                normaliser_texte(
                    candidat["texte"]
                )
                and
                abs(
                    mot["x"]
                    -
                    candidat["x"]
                )
                <=
                25
                and
                abs(
                    mot["y"]
                    -
                    candidat["y"]
                )
                <=
                25
            ):

                ancien = candidat
                break

        if ancien is None:

            resultat.append(
                mot
            )

        elif (
            mot["confiance"]
            >
            ancien["confiance"]
        ):

            ancien.update(
                mot
            )

    return resultat


def detecter_lignes_depuis_ocr(
    mots_panel
):

    """
    Détecte les lignes à partir des mots OCR contenant un nom
    d'unité connu. Ce fallback est utilisé quand les barres vertes
    ne sont plus vertes parce que les soins ont déjà été lancés.
    """

    if not mots_panel:
        return []

    candidats = []

    for mot in mots_panel:

        texte = normaliser_texte(
            mot["texte"]
        )

        if not texte:
            continue

        # On cherche le début des noms d'unités dans les mots OCR.
        # Même si le nom est composé de plusieurs mots, un mot
        # suffisamment distinctif peut servir à repérer sa ligne.
        for nom_normalise in UNIT_NAMES_SORTED:

            morceaux = nom_normalise.split()

            if (
                texte == morceaux[0]
                or texte in morceaux
                or morceaux[0] in texte
            ):

                candidats.append(
                    mot["y"]
                )

                break

    if not candidats:
        return []

    candidats.sort()

    lignes = []

    for y in candidats:

        if not lignes:

            lignes.append(y)
            continue

        if abs(y - lignes[-1]) <= 22:

            lignes[-1] = (
                lignes[-1] + y
            ) // 2

        else:

            lignes.append(y)

    return lignes


# =========================================================
# TROUVER UNITE
# =========================================================

def trouver_unite(
    mots
):

    if not mots:

        return None, None

    texte = " ".join(
        mot["texte"]
        for mot in mots
    )

    normalise = normaliser_texte(
        texte
    )

    for nom_normalise in UNIT_NAMES_SORTED:

        if nom_normalise in normalise:

            return (
                nom_normalise,
                UNIT_TIERS[nom_normalise]
            )

    return None, None


# =========================================================
# TROUVER TIER
# =========================================================

def trouver_tier(
    mots
):

    _, tier = trouver_unite(
        mots
    )

    return tier


# =========================================================
# EXTRAIRE CANDIDATS NUMERIQUES
# =========================================================

def extraire_candidats_numeriques(
    mots
):

    candidats = []

    for mot in mots:

        brut = (
            mot["texte"]
            .replace(
                ",",
                ""
            )
            .replace(
                ".",
                ""
            )
            .replace(
                " ",
                ""
            )
        )

        if not re.fullmatch(
            r"\d+",
            brut
        ):
            continue

        try:

            valeur = int(
                brut
            )

        except ValueError:

            continue

        if not (
            0
            <=
            valeur
            <=
            999999999
        ):
            continue

        candidats.append(
            {
                "x":
                    mot["x"],

                "y":
                    mot["y"],

                "valeur":
                    valeur
            }
        )

    return candidats


# =========================================================
# TROUVER NOMBRE
# =========================================================

def trouver_nombre(
    mots,
    nom_unite,
    image_width
):

    """
    Trouve la quantité de la ligne en se basant sur la position
    réelle du NOM de l'unité.

    Le problème précédent était que la fenêtre verticale pouvait
    contenir plusieurs nombres, notamment le total des blessés.
    Ici, la quantité doit :
      1. être proche verticalement du nom de l'unité ;
      2. être située à droite du nom ;
      3. être dans le panneau des unités.

    Cela évite de récupérer 280004 depuis "Blessés graves".
    """

    if not mots or not nom_unite:
        return None

    candidats = extraire_candidats_numeriques(
        mots
    )

    if not candidats:
        return None

    nom_morceaux = nom_unite.split()

    mots_nom = []

    for mot in mots:

        mot_normalise = normaliser_texte(
            mot["texte"]
        )

        if mot_normalise in nom_morceaux:

            mots_nom.append(
                mot
            )

    # Si OCR a regroupé plusieurs mots du nom dans un seul token,
    # on retrouve quand même un point de référence avec le premier mot.
    if not mots_nom:

        premier = nom_morceaux[0]

        for mot in mots:

            mot_normalise = normaliser_texte(
                mot["texte"]
            )

            if (
                premier in mot_normalise
                or
                mot_normalise in premier
            ):

                mots_nom.append(
                    mot
                )

    if not mots_nom:
        return None

    nom_y = sum(
        mot["y"]
        for mot in mots_nom
    ) / len(mots_nom)

    nom_x_max = max(
        mot["x"]
        for mot in mots_nom
    )

    # Quantité sur la même ligne et à droite du nom.
    candidats_ligne = [
        candidat
        for candidat in candidats
        if (
            abs(
                candidat["y"]
                -
                nom_y
            )
            <=
            35
        )
        and
        (
            candidat["x"]
            >
            nom_x_max
        )
        and
        (
            candidat["x"] / image_width
            >=
            0.55
        )
    ]

    if candidats_ligne:

        return max(
            candidats_ligne,
            key=lambda candidat:
            candidat["x"]
        )["valeur"]

    # Second essai : certains OCR placent le point gauche du nombre
    # légèrement avant la fin du dernier mot du nom.
    candidats_secours = [
        candidat
        for candidat in candidats
        if (
            abs(
                candidat["y"]
                -
                nom_y
            )
            <=
            35
        )
        and
        (
            candidat["x"] / image_width
            >=
            0.55
        )
    ]

    if candidats_secours:

        return max(
            candidats_secours,
            key=lambda candidat:
            candidat["x"]
        )["valeur"]

    return None


# =========================================================
# ANALYSER LIGNE
# =========================================================

def analyser_ligne(
    row_y,
    mots_panel,
    image_width
):

    # Fenêtre suffisamment large pour retrouver le nom complet,
    # mais la quantité sera ensuite liée verticalement au nom.
    mots = [
        mot
        for mot in mots_panel
        if (
            row_y - 65
            <=
            mot["y"]
            <=
            row_y + 25
        )
    ]

    nom_unite, tier = trouver_unite(
        mots
    )

    nombre = trouver_nombre(
        mots,
        nom_unite,
        image_width
    )

    return (
        nom_unite,
        tier,
        nombre
    )


# =========================================================
# CONVERTIR RESSOURCE
# =========================================================

def convertir_ressource(
    texte
):

    if not texte:

        return None

    texte = (
        texte
        .upper()
        .replace(
            ",",
            "."
        )
        .replace(
            " ",
            ""
        )
    )

    match = re.search(
        r"(\d+(?:\.\d+)?)(K|M|B)?",
        texte
    )

    if not match:

        return None

    try:

        valeur = float(
            match.group(1)
        )

    except ValueError:

        return None

    unite = match.group(2)

    if unite == "K":

        valeur *= 1000

    elif unite == "M":

        valeur *= 1000000

    elif unite == "B":

        valeur *= 1000000000

    return int(
        round(
            valeur
        )
    )


# =========================================================
# ANALYSER TEXTE RESSOURCE TOKEN
# =========================================================

def analyser_texte_ressource_token(
    texte
):
    """
    Transforme un token OCR représentant une ressource en entier.

    Exemples :
    33.1M -> 33100000
    10.4M -> 10400000
    2.1K  -> 2100
    144   -> 144
    """

    if not texte:
        return None

    propre = (
        texte
        .upper()
        .replace(",", ".")
        .replace(" ", "")
    )

    # Ignore les timers et autres textes contenant ':'.
    if ":" in propre:
        return None

    return convertir_ressource(
        propre
    )


# =========================================================
#  TROUVER ICONES RESSOURCES
# =========================================================

def _trouver_icones_ressources(
    image
):
    """
    Détecte les icônes de ressources sur PC et téléphone.

    Signatures :
    - food  : vert + jaune
    - wood  : orange/brun
    - stone : gros motif gris clair/bleuté
    - gold  : jaune/orange sans vert

    Le but est de ne jamais supposer que 3 ressources signifie
    automatiquement Wood/Stone/Gold : le jeu peut afficher
    Food/Wood/Gold, par exemple.
    """

    h, w = image.shape[:2]

    # La barre est toujours dans la partie droite et basse du panneau.
    x1 = int(w * 0.34)
    x2 = int(w * 0.995)
    y1 = int(h * 0.68)
    y2 = int(h * 0.94)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    H, S, V = cv2.split(hsv)

    def components(
        mask,
        min_area,
        max_area=1200
    ):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            8
        )

        result = []

        for i in range(1, count):

            area = int(
                stats[i, cv2.CC_STAT_AREA]
            )

            if not (
                min_area
                <=
                area
                <=
                max_area
            ):
                continue

            width = int(
                stats[i, cv2.CC_STAT_WIDTH]
            )

            height = int(
                stats[i, cv2.CC_STAT_HEIGHT]
            )

            if width < 5 or height < 5:
                continue

            cx, cy = centers[i]

            result.append(
                {
                    "x":
                        float(cx + x1),

                    "y":
                        float(cy + y1),

                    "area":
                        area,

                    "width":
                        width,

                    "height":
                        height
                }
            )

        return result

    # Food leaves.
    food_mask = (
        (H >= 35)
        &
        (H < 90)
        &
        (S > 110)
        &
        (V > 55)
    )

    # Wood log.
    wood_mask = (
        (H < 25)
        &
        (S > 100)
        &
        (V > 55)
    )

    # Gold coin / food grain highlights.
    yellow_mask = (
        (H >= 20)
        &
        (H < 42)
        &
        (S > 100)
        &
        (V > 70)
    )

    # Stone is grey/light-blue, so saturation is lower.
    stone_mask = (
        (S < 120)
        &
        (V > 125)
    )

    foods = components(
        food_mask,
        25
    )

    woods = components(
        wood_mask,
        60
    )

    yellows = components(
        yellow_mask,
        45
    )

    stones = components(
        stone_mask,
        120
    )

    icons = []

    def add_icon(
        x,
        y,
        resource
    ):
        # Merge close detections belonging to the same icon.
        for icon in icons:

            if (
                abs(
                    icon["x"]
                    -
                    x
                )
                <=
                22
                and
                abs(
                    icon["y"]
                    -
                    y
                )
                <=
                22
            ):

                # Food is the most distinctive, then wood, stone, gold.
                priority = {
                    "food":
                        4,

                    "wood":
                        3,

                    "stone":
                        2,

                    "gold":
                        1
                }

                if (
                    priority[resource]
                    >
                    priority[icon["resource"]]
                ):

                    icon["resource"] = resource

                return

        icons.append(
            {
                "x":
                    x,

                "y":
                    y,

                "resource":
                    resource
            }
        )

    # Food: green is unique enough that it can be used directly.
    for item in foods:

        add_icon(
            item["x"],
            item["y"],
            "food"
        )

    # Gold: yellow not associated with food.
    for item in yellows:

        near_food = any(
            icon["resource"] == "food"
            and
            abs(
                icon["x"]
                -
                item["x"]
            )
            <=
            24
            and
            abs(
                icon["y"]
                -
                item["y"]
            )
            <=
            24
            for icon in icons
        )

        if not near_food:

            add_icon(
                item["x"],
                item["y"],
                "gold"
            )

    # Wood: orange not already part of a gold coin.
    for item in woods:

        near_gold = any(
            icon["resource"] == "gold"
            and
            abs(
                icon["x"]
                -
                item["x"]
            )
            <=
            24
            and
            abs(
                icon["y"]
                -
                item["y"]
            )
            <=
            24
            for icon in icons
        )

        if not near_gold:

            add_icon(
                item["x"],
                item["y"],
                "wood"
            )

    # Stone: keep only reasonably compact large grey blobs.
    for item in stones:

        if (
            item["width"] >= 10
            and
            item["height"] >= 10
        ):

            add_icon(
                item["x"],
                item["y"],
                "stone"
            )

    # A few OCR/background components may still survive; keep icons
    # ordered and merge very close duplicates.
    icons.sort(
        key=lambda icon:
        icon["x"]
    )

    cleaned = []

    for icon in icons:

        if not cleaned:

            cleaned.append(
                icon
            )

            continue

        previous = cleaned[-1]

        if (
            abs(
                icon["x"]
                -
                previous["x"]
            )
            <=
            30
        ):

            priority = {
                "food":
                    4,

                "wood":
                    3,

                "stone":
                    2,

                "gold":
                    1
            }

            if (
                priority[icon["resource"]]
                >
                priority[previous["resource"]]
            ):

                cleaned[-1] = icon

        else:

            cleaned.append(
                icon
            )

    return cleaned


# =========================================================
#  LIRE MONTANT A COTE ICONE
# =========================================================

def _lire_montant_a_cote_icone(
    image,
    icon,
    autres_icones
):
    """
    Lit le montant immédiatement à droite de l'icône.
    """

    h, w = image.shape[:2]

    x = float(icon["x"])
    y = float(icon["y"])

    suivantes = sorted(
        autre["x"]
        for autre in autres_icones
        if autre["x"] > x
    )

    if suivantes:

        droite = min(
            suivantes[0] - 6,
            x + max(
                120,
                int(w * 0.22)
            )
        )

    else:

        droite = min(
            w - 3,
            x + max(
                160,
                int(w * 0.25)
            )
        )

    gauche = int(
        min(
            w - 2,
            x + max(
                6,
                int(w * 0.008)
            )
        )
    )

    droite = int(
        max(
            gauche + 20,
            droite
        )
    )

    demi = max(
        18,
        int(h * 0.045)
    )

    haut = max(
        0,
        int(y - demi)
    )

    bas = min(
        h,
        int(y + demi)
    )

    crop = image[
        haut:bas,
        gauche:droite
    ]

    if crop.size == 0:
        return None

    valeurs = []

    for scale in (
        4,
        6,
        8,
        10
    ):

        agrandi = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            agrandi,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        variantes = [
            gray,
            clahe.apply(gray)
        ]

        for seuil in (
            80,
            110,
            140,
            170,
            200
        ):

            _, b = cv2.threshold(
                gray,
                seuil,
                255,
                cv2.THRESH_BINARY
            )

            variantes.append(
                b
            )

        for variante in variantes:

            for psm in (
                6,
                7,
                11,
                13
            ):

                try:

                    texte = pytesseract.image_to_string(
                        variante,
                        config=(
                            f"--psm {psm} "
                            "-c tessedit_char_whitelist="
                            "0123456789.KMB"
                        )
                    )

                except Exception:

                    continue

                texte = (
                    texte
                    .upper()
                    .replace(",", ".")
                    .strip()
                )

                # Montants complets.
                for match in re.findall(
                    r"\d+(?:\.\d+)?\s*[KMB]",
                    texte
                ):

                    valeur = convertir_ressource(
                        match
                    )

                    if (
                        valeur is not None
                        and
                        valeur > 0
                    ):

                        valeurs.append(
                            valeur
                        )

                # Gold peut ne pas avoir de suffixe.
                if icon["resource"] == "gold":

                    morceaux = re.findall(
                        r"\b\d{1,5}\b",
                        texte
                    )

                    for morceau in morceaux:

                        valeur = int(
                            morceau
                        )

                        if (
                            1
                            <=
                            valeur
                            <=
                            99999
                        ):

                            valeurs.append(
                                valeur
                            )

    if not valeurs:
        return None

    compte = Counter(
        valeurs
    )

    return compte.most_common(
        1
    )[0][0]


def _analyser_ressources_legacy(
    image
):
    """
    Analyse les ressources de la fenêtre d'hôpital.

    L'ordre des ressources n'est jamais supposé.
    Le type est donné par l'icône détectée à gauche de chaque montant.

    Cela gère notamment :
        Food / Wood / Gold
        Wood / Stone / Gold
        Food / Wood / Stone / Gold
    """

    icones = _trouver_icones_ressources(
        image
    )

    if not icones:

        print()
        print(
            "Ressources : aucune icône détectée."
        )

        return {
            "nourriture":
                None,

            "bois":
                None,

            "pierre":
                None,

            "or":
                None
        }

    # Trier les icônes de gauche à droite.
    icones = sorted(
        icones,
        key=lambda icon:
        icon["x"]
    )

    resultats = {
        "food":
            None,

        "wood":
            None,

        "stone":
            None,

        "gold":
            None
    }

    for icon in icones:

        valeur = _lire_montant_a_cote_icone(
            image,
            icon,
            icones
        )

        if valeur is None:
            continue

        # Une seule valeur par type.
        # En cas de doublon, garder la première lecture valide.
        if resultats[
            icon["resource"]
        ] is None:

            resultats[
                icon["resource"]
            ] = valeur

    print()
    print(
        "Ressources identifiées par icône :"
    )

    print(
        "Nourriture :",
        resultats["food"]
    )

    print(
        "Bois       :",
        resultats["wood"]
    )

    print(
        "Pierre     :",
        resultats["stone"]
    )

    print(
        "Or         :",
        resultats["gold"]
    )

    return {
        "nourriture":
            resultats["food"],

        "bois":
            resultats["wood"],

        "pierre":
            resultats["stone"],

        "or":
            resultats["gold"]
    }





def _ocr_pc_value(
    image,
    x1,
    y1,
    x2,
    y2
):
    """
    OCR spécialisé d'une valeur PC de type 33.1M / 10.4M / 17.7M / 2.2M.
    """

    h, w = image.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    observations = []

    for scale in (
        4,
        5
    ):

        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY
        )

        for variant in (
            enlarged,
            gray
        ):

            try:

                ocr = pytesseract.image_to_string(
                    variant,
                    config=(
                        "--psm 7 "
                        "-c tessedit_char_whitelist="
                        "0123456789.KMB"
                    )
                )

            except Exception:

                continue

            ocr = (
                ocr
                .upper()
                .replace(
                    ",",
                    "."
                )
                .replace(
                    " ",
                    ""
                )
                .replace(
                    "\n",
                    ""
                )
                .strip()
            )

            for token in re.findall(
                r"\d+(?:\.\d+)?[KMB]",
                ocr
            ):

                value = convertir_ressource(
                    token
                )

                if (
                    value is not None
                    and
                    value > 0
                ):

                    observations.append(
                        (
                            value,
                            token
                        )
                    )

    if not observations:
        return None

    # Prefer exact decimal readings (33.1M instead of 331M, etc.).
    decimal_values = [
        value
        for value, token in observations
        if "." in token
    ]

    if decimal_values:

        return Counter(
            decimal_values
        ).most_common(
            1
        )[0][0]

    return Counter(
        value
        for value, _ in observations
    ).most_common(
        1
    )[0][0]


def _ocr_gold_pc(
    image,
    x_icon,
    y_icon,
    x_right
):
    """
    Lecteur dédié à l'or du layout PC.

    L'icône de l'or est juste à gauche du montant 2.2M. Sur certaines
    captures, Tesseract interprète un bord de l'icône comme un "1" et
    retourne 12.2M. On effectue donc plusieurs coupes horizontales
    légèrement décalées et on ne garde que les valeurs corroborées.
    """

    h, w = image.shape[:2]

    valeurs = []

    demi = max(
        18,
        int(
            h * 0.035
        )
    )

    y1 = max(
        0,
        int(
            y_icon - demi
        )
    )

    y2 = min(
        h,
        int(
            y_icon + demi
        )
    )

    # Plusieurs offsets : ils doivent tous lire le même montant si
    # le texte est réellement présent.
    for offset in (
        8,
        12,
        16,
        20,
        24
    ):

        x1 = max(
            0,
            int(
                x_icon + offset
            )
        )

        x2 = min(
            w - 2,
            int(
                x_right
            )
        )

        if x2 <= x1:
            continue

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        for scale in (
            4,
            5,
            6
        ):

            enlarged = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(
                enlarged,
                cv2.COLOR_BGR2GRAY
            )

            variants = [
                enlarged,
                gray
            ]

            for threshold in (
                120,
                160,
                200
            ):

                _, binary = cv2.threshold(
                    gray,
                    threshold,
                    255,
                    cv2.THRESH_BINARY
                )

                variants.append(
                    binary
                )

            for variant in variants:

                for psm in (
                    6,
                    7,
                    11
                ):

                    try:

                        ocr = pytesseract.image_to_string(
                            variant,
                            config=(
                                f"--psm {psm} "
                                "-c tessedit_char_whitelist="
                                "0123456789.KMB"
                            )
                        )

                    except Exception:

                        continue

                    ocr = (
                        ocr
                        .upper()
                        .replace(
                            ",",
                            "."
                        )
                        .replace(
                            " ",
                            ""
                        )
                        .replace(
                            "\n",
                            ""
                        )
                        .strip()
                    )

                    for token in re.findall(
                        r"\d+(?:\.\d+)?[KMB]",
                        ocr
                    ):

                        value = convertir_ressource(
                            token
                        )

                        if (
                            value is not None
                            and
                            value > 0
                        ):

                            valeurs.append(
                                value
                            )

    if not valeurs:
        return None

    # Vote exact. Une valeur réellement présente devrait survivre à
    # plusieurs offsets indépendants.
    compte = Counter(
        valeurs
    )

    meilleure, votes = (
        compte.most_common(
            1
        )[0]
    )

    # Si 12.2M apparaît une fois mais 2.2M plusieurs fois, 2.2M gagne.
    return meilleure



def _analyser_ressources_pc(
    image
):
    """
    Layout PC : les quatre ressources occupent quatre slots fixes.

    On ne fait donc pas deviner le type par OCR :
        slot 1 = Food
        slot 2 = Wood
        slot 3 = Stone
        slot 4 = Gold

    Cette méthode est utilisée uniquement si au moins trois slots sont
    effectivement lisibles comme des montants K/M/B.
    """

    h, w = image.shape[:2]

    slots = [
        ("food", 0.445),
        ("wood", 0.590),
        ("stone", 0.735),
        ("gold", 0.865),
    ]

    positions = [
        int(
            w * ratio
        )
        for _, ratio in slots
    ]

    result = {
        "food":
            None,

        "wood":
            None,

        "stone":
            None,

        "gold":
            None
    }

    lus = 0

    y = int(
        h * 0.81
    )

    for index, (resource, _) in enumerate(
        slots
    ):

        x = positions[index]

        if index + 1 < len(
            positions
        ):

            right = (
                positions[index + 1]
                -
                max(
                    8,
                    int(
                        w * 0.006
                    )
                )
            )

        else:

            right = min(
                w - 3,
                x
                +
                int(
                    w * 0.15
                )
            )

        # Important : +8 preserve le premier chiffre de 17.7M.
        if resource == "gold":

            value = _ocr_gold_pc(
                image,
                x,
                y,
                right
            )

        else:

            value = _ocr_pc_value(
                image,
                x + 8,
                y - int(h * 0.035),
                right,
                y + int(h * 0.035)
            )

        if value is not None:

            result[
                resource
            ] = value

            lus += 1

    # Si quatre montants sont trouvés, c'est quasi certainement le layout PC.
    # Trois suffisent quand une ressource est temporairement illisible.
    if lus < 3:
        return None

    return {
        "nourriture":
            result["food"],

        "bois":
            result["wood"],

        "pierre":
            result["stone"],

        "or":
            result["gold"]
    }


def _analyser_ressources_legacy(
    image
):
    """
    Essaie d'abord le lecteur PC à quatre slots.
    Si la capture n'est pas au format PC, l'ancien lecteur téléphone
    reste utilisé sans modification.
    """

    pc = _analyser_ressources_pc(
        image
    )

    if pc is not None:

        print()
        print(
            "Ressources PC :"
        )

        print(
            "Nourriture :",
            pc["nourriture"]
        )

        print(
            "Bois       :",
            pc["bois"]
        )

        print(
            "Pierre     :",
            pc["pierre"]
        )

        print(
            "Or         :",
            pc["or"]
        )

        return pc

    return _analyser_ressources_legacy(
        image
    )





def _ocr_phone_plain_amount(
    image,
    x1,
    y1,
    x2,
    y2
):
    """
    Lit UN montant du layout téléphone.

    Important : 432, 324 et 29 sont des nombres simples. Ils ne doivent
    pas être convertis en 432000/324000. Une conversion K/M/B n'est faite
    que lorsqu'un suffixe est réellement présent.
    """

    h, w = image.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    observations = []

    # Pour les screenshots normalisés, quelques passes suffisent et évitent
    # de multiplier inutilement les appels Tesseract.
    for scale in (
        4,
        6,
        8
    ):

        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY
        )

        variants = [
            gray
        ]

        # Contraste local.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        variants.append(
            clahe.apply(
                gray
            )
        )

        # Le texte clair du jeu est bien séparé par ces seuils.
        for threshold in (
            110,
            150,
            190
        ):

            _, binary = cv2.threshold(
                gray,
                threshold,
                255,
                cv2.THRESH_BINARY
            )

            variants.append(
                binary
            )

        for variant in variants:

            for psm in (
                6,
                7
            ):

                try:

                    data = pytesseract.image_to_data(
                        variant,
                        config=(
                            f"--psm {psm} "
                            "-c tessedit_char_whitelist="
                            "0123456789.KMB"
                        ),
                        output_type=pytesseract.Output.DICT
                    )

                except Exception:

                    continue

                tokens = [
                    token.strip().upper()
                    for token in data["text"]
                    if token.strip()
                ]

                # Montants complets.
                for token in tokens:

                    token = token.replace(
                        ",",
                        "."
                    )

                    if re.fullmatch(
                        r"\d+(?:\.\d+)?[KMB]",
                        token
                    ):

                        value = convertir_ressource(
                            token
                        )

                        if (
                            value is not None
                            and
                            value > 0
                        ):

                            observations.append(
                                value
                            )

                    elif re.fullmatch(
                        r"\d{1,6}",
                        token
                    ):

                        value = int(
                            token
                        )

                        if (
                            0 < value <= 999999
                        ):

                            observations.append(
                                value
                            )

                # OCR peut découper un nombre en plusieurs tokens.
                for i in range(
                    len(tokens) - 1
                ):

                    a = (
                        tokens[i]
                        .replace(",", ".")
                    )

                    b = (
                        tokens[i + 1]
                        .replace(",", ".")
                    )

                    fusion = (
                        a
                        +
                        b
                    )

                    if re.fullmatch(
                        r"\d{2,6}",
                        fusion
                    ):

                        value = int(
                            fusion
                        )

                        if (
                            0 < value <= 999999
                        ):

                            observations.append(
                                value
                            )

    if not observations:
        return None

    # Vote majoritaire.
    counts = Counter(
        observations
    )

    return counts.most_common(
        1
    )[0][0]


def _analyser_ressources_phone(
    image
):
    """
    Analyse le layout téléphone à partir des vraies icônes détectées.

    Les trois slots sont identifiés par leur position, mais leur TYPE est
    donné par l'icône. On ne suppose donc jamais :
        slot 1 = Food
        slot 2 = Wood
        slot 3 = Gold

    Cela permet notamment :
        Food / Wood / Gold
        Food / Stone / Gold
        Food / Wood / Stone
        etc.
    """

    h, w = image.shape[:2]

    icons = _trouver_icones_ressources(
        image
    )

    # Garder uniquement les icônes de la ligne de ressources.
    # Les faux composants des compteurs de l'hôpital sont plus hauts.
    icons = [
        icon
        for icon in icons
        if (
            icon["x"] >= w * 0.40
            and
            icon["y"] >= h * 0.70
            and
            icon["y"] <= h * 0.92
        )
    ]

    # Dédupliquer par proximité spatiale.
    icons = sorted(
        icons,
        key=lambda icon:
        icon["x"]
    )

    uniques = []

    for icon in icons:

        proche = None

        for ancien in uniques:

            if (
                abs(
                    icon["x"]
                    -
                    ancien["x"]
                )
                <=
                35
                and
                abs(
                    icon["y"]
                    -
                    ancien["y"]
                )
                <=
                30
            ):

                proche = ancien
                break

        if proche is None:
            uniques.append(
                icon
            )

    icons = uniques

    # Le layout téléphone doit présenter au moins deux ressources.
    if len(icons) < 2:
        return None

    result = {
        "food":
            None,

        "wood":
            None,

        "stone":
            None,

        "gold":
            None
    }

    for index, icon in enumerate(
        icons
    ):

        x = float(
            icon["x"]
        )

        y = float(
            icon["y"]
        )

        # Limite droite = juste avant l'icône suivante.
        if index + 1 < len(icons):

            right = (
                icons[index + 1]["x"]
                -
                max(
                    8,
                    int(
                        w * 0.008
                    )
                )
            )

        else:

            right = min(
                w - 3,
                x
                +
                int(
                    w * 0.16
                )
            )

        half = max(
            16,
            int(
                h * 0.045
            )
        )

        value = _ocr_phone_plain_amount(
            image,
            x + max(
                3,
                int(
                    w * 0.004
                )
            ),
            y - half,
            right,
            y + half
        )

        if value is None:
            continue

        resource = icon[
            "resource"
        ]

        if result[
            resource
        ] is None:

            result[
                resource
            ] = value

    # Éviter de confondre un layout PC incomplet avec le téléphone.
    # Au moins deux ressources lisibles sont nécessaires.
    if sum(
        value is not None
        for value in result.values()
    ) < 2:

        return None

    return {
        "nourriture":
            result["food"],

        "bois":
            result["wood"],

        "pierre":
            result["stone"],

        "or":
            result["gold"]
    }


def analyser_ressources(
    image
):
    """
    Ordre de détection :
        1. Téléphone : icônes réelles + nombres simples exacts.
        2. PC : lecteur spécialisé à quatre slots.
        3. Ancien fallback, uniquement si aucune des deux méthodes
           précédentes n'est suffisamment fiable.
    """

    phone = _analyser_ressources_phone(
        image
    )

    if phone is not None:

        print()
        print(
            "Ressources téléphone :"
        )

        print(
            "Nourriture :",
            phone["nourriture"]
        )

        print(
            "Bois       :",
            phone["bois"]
        )

        print(
            "Pierre     :",
            phone["pierre"]
        )

        print(
            "Or         :",
            phone["or"]
        )

        return phone

    pc = _analyser_ressources_pc(
        image
    )

    if pc is not None:
        return pc

    return _analyser_ressources_legacy(
        image
    )


# =========================================================
# ISOLER FENETRE HOPITAL
# =========================================================

def isoler_fenetre_hopital(
    image
):
    """
    Isole la fenêtre complète de l'hôpital.

    PRIORITE :
    1) détecter le bandeau beige du titre "SOIN DES UNITÉS" ;
    2) reconstruire les limites de la fenêtre à partir de ce bandeau.

    C'est beaucoup plus fiable qu'un simple gros contour bleu lorsque
    la capture contient la ville entière autour de l'hôpital.

    Ensuite, tout l'OCR travaille uniquement dans cette fenêtre.
    """

    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # ---------------------------------------------------------
    # 1. Chercher le bandeau beige de la fenêtre d'hôpital.
    # ---------------------------------------------------------
    #
    # Le bandeau est très clair, peu saturé et extrêmement horizontal.
    # Les éléments lumineux de la ville sont beaucoup plus petits.
    # ---------------------------------------------------------

    beige_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 135]),
        np.array([179, 120, 255])
    )

    beige_mask = cv2.morphologyEx(
        beige_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    15,
                    int(w * 0.012)
                ),
                max(
                    7,
                    int(h * 0.008)
                )
            )
        )
    )

    contours, _ = cv2.findContours(
        beige_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    title_candidates = []

    for contour in contours:

        x, y, ww, hh = cv2.boundingRect(
            contour
        )

        if ww < w * 0.35:
            continue

        if hh < h * 0.02:
            continue

        if hh > h * 0.16:
            continue

        ratio = ww / max(
            hh,
            1
        )

        if ratio < 7:
            continue

        # Une vraie fenêtre est proche du centre de la capture.
        cx = x + ww / 2
        center_distance = abs(
            cx - w / 2
        ) / max(
            w,
            1
        )

        score = (
            (ww / w) * 100
            -
            center_distance * 25
        )

        title_candidates.append(
            (
                score,
                x,
                y,
                ww,
                hh
            )
        )

    if title_candidates:

        title_candidates.sort(
            reverse=True
        )

        _, tx, ty, tw, th = (
            title_candidates[0]
        )

        # Le bandeau détecté correspond à l'intérieur de la fenêtre.
        # On reconstruit le cadre complet autour.
        left = max(
            0,
            int(
                tx
                -
                tw * 0.085
            )
        )

        right = min(
            w,
            int(
                tx
                +
                tw * 1.005
            )
        )

        top = max(
            0,
            int(
                ty
                -
                th * 0.82
            )
        )

        bottom = min(
            h,
            int(
                ty
                +
                th
                +
                tw * 0.515
            )
        )

        # Vérifier que les proportions correspondent à une fenêtre
        # d'hôpital plausible.
        crop_w = right - left
        crop_h = bottom - top

        if (
            crop_w > w * 0.35
            and
            crop_h > h * 0.30
            and
            1.30
            <=
            crop_w / max(crop_h, 1)
            <=
            3.0
        ):

            crop = image[
                top:bottom,
                left:right
            ]

            if crop.size:

                print(
                    f"Fenêtre hôpital isolée par titre : "
                    f"x={left}:{right}, "
                    f"y={top}:{bottom}"
                )

                return crop

    # ---------------------------------------------------------
    # 2. FALLBACK : ancien détecteur bleu.
    # ---------------------------------------------------------

    blue_mask = cv2.inRange(
        hsv,
        np.array([80, 55, 35]),
        np.array([130, 255, 255])
    )

    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    7,
                    int(w * 0.008)
                ),
                max(
                    7,
                    int(h * 0.008)
                )
            )
        )
    )

    blue_contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidats = []

    image_cx = w / 2
    image_cy = h / 2

    for contour in blue_contours:

        x, y, ww, hh = cv2.boundingRect(
            contour
        )

        area_ratio = (
            ww * hh
        ) / float(
            w * h
        )

        if area_ratio < 0.12:
            continue

        if ww < w * 0.35:
            continue

        if hh < h * 0.25:
            continue

        ratio = ww / max(
            hh,
            1
        )

        if not (
            1.15
            <=
            ratio
            <=
            3.2
        ):
            continue

        cx = x + ww / 2
        cy = y + hh / 2

        distance = (
            (
                (cx - image_cx)
                /
                w
            ) ** 2
            +
            (
                (cy - image_cy)
                /
                h
            ) ** 2
        ) ** 0.5

        score = (
            area_ratio * 10
            -
            distance * 2
        )

        candidats.append(
            (
                score,
                x,
                y,
                ww,
                hh
            )
        )

    if not candidats:

        print(
            "Fenêtre hôpital : "
            "aucune fenêtre détectée, "
            "image complète utilisée."
        )

        return image

    candidats.sort(
        reverse=True
    )

    _, x, y, ww, hh = candidats[0]

    left = max(
        0,
        x - int(ww * 0.10)
    )

    right = min(
        w,
        x + ww + int(ww * 0.10)
    )

    top = max(
        0,
        y - int(hh * 0.20)
    )

    bottom = min(
        h,
        y + hh + int(hh * 0.10)
    )

    crop = image[
        top:bottom,
        left:right
    ]

    if crop.size == 0:
        return image

    print(
        f"Fenêtre hôpital isolée par bleu : "
        f"x={left}:{right}, "
        f"y={top}:{bottom}"
    )

    return crop


# =========================================================
# ANALYSER IMAGE
# =========================================================


def normaliser_resolution_ocr(
    image,
    largeur_cible=1800
):
    """
    Agrandit automatiquement les petites captures avant toute détection.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    if w >= largeur_cible:
        return image

    facteur = min(
        largeur_cible / float(w),
        4.0
    )

    return cv2.resize(
        image,
        (
            int(round(w * facteur)),
            int(round(h * facteur))
        ),
        interpolation=cv2.INTER_CUBIC
    )




def analyser_image(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Impossible d'ouvrir : "
            f"{image_path}"
        )

    image = normaliser_resolution_ocr(
        image,
        largeur_cible=1800
    )

    print(
        f"Résolution source normalisée : "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    image = isoler_fenetre_hopital(
        image
    )

    print(
        f"Résolution fenêtre hôpital : "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    print(
        "========================================"
    )

    print(
        f"IMAGE : {image_path}"
    )

    print(
        "========================================"
    )

    # =====================================================
    # TROUPES
    # =====================================================

    lignes_vertes = detecter_lignes(
        image
    )

    mots_panel = ocr_panneau_troupes(
        image
    )

    # Quand les soins sont déjà lancés, les barres ne sont plus
    # vertes et detecter_lignes() peut retourner 0 ligne.
    # On récupère alors les positions directement depuis les noms
    # d'unités reconnus par l'OCR.
    lignes_ocr = detecter_lignes_depuis_ocr(
        mots_panel
    )

    lignes = list(lignes_vertes)

    for y in lignes_ocr:

        proche = any(
            abs(y - existant) <= 35
            for existant in lignes
        )

        if not proche:
            lignes.append(y)

    lignes.sort()

    print(
        f"{len(lignes)} ligne(s) détectée(s) "
        f"(barres: {len(lignes_vertes)}, OCR: {len(lignes_ocr)})"
    )

    troupes = []

    lignes_illisibles = []

    for row_y in lignes:

        nom_unite, tier, nombre = analyser_ligne(
            row_y,
            mots_panel,
            image.shape[1]
        )

        if tier is None:

            print(
                f"Ligne {row_y} : "
                "unité indéterminée"
            )

            continue

        if nombre is None:

            print(
                f"Ligne {row_y} : "
                f"{nom_unite} ({tier}) = "
                "nombre illisible"
            )

            lignes_illisibles.append(
                {
                    "nom":
                        nom_unite,

                    "type":
                        tier,

                    "row_y":
                        row_y
                }
            )

            continue

        print(
            f"Ligne {row_y} : "
            f"{nom_unite} = "
            f"{tier} = {nombre}"
        )

        troupes.append(
            {
                "nom":
                    nom_unite,

                "type":
                    tier,

                "nombre":
                    nombre
            }
        )

    # =====================================================
    # RESSOURCES
    # =====================================================

    ressources = analyser_ressources(
        image
    )

    return {

        "troupes":
            troupes,

        "lignes_illisibles":
            lignes_illisibles,

        "nourriture":
            ressources[
                "nourriture"
            ],

        "bois":
            ressources[
                "bois"
            ],

        "pierre":
            ressources[
                "pierre"
            ],

        "or":
            ressources[
                "or"
            ]
    }


# =========================================================
# EST DOUBLON
# =========================================================

def est_doublon(
    a,
    b
):

    # IMPORTANT :
    # le nombre seul ne permet PAS d'identifier une unité.
    # Exemple : Chevalier 3 et Maryannu 3 sont deux lignes
    # différentes et doivent toutes les deux être conservées.
    #
    # Une ligne identique sur les deux captures est un doublon
    # seulement si le NOM + TIER + QUANTITE correspondent.

    return (
        a.get("nom")
        ==
        b.get("nom")
        and
        a["type"]
        ==
        b["type"]
        and
        a["nombre"]
        ==
        b["nombre"]
    )


# =========================================================
# ANALYSER PLUSIEURS IMAGES
# =========================================================

def analyser_plusieurs_images(
    images
):

    if not images:

        raise ValueError(
            "Aucune image fournie."
        )

    toutes_les_troupes = []

    toutes_les_lignes_illisibles = []

    resultats_images = []

    # =====================================================
    # ANALYSE DES IMAGES
    # =====================================================

    for image_path in images:

        resultat = analyser_image(
            image_path
        )

        resultats_images.append(
            resultat
        )

        toutes_les_troupes.extend(
            resultat[
                "troupes"
            ]
        )

        toutes_les_lignes_illisibles.extend(
            resultat[
                "lignes_illisibles"
            ]
        )

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    troupes_finales = []

    for troupe in toutes_les_troupes:

        doublon = False

        for existant in troupes_finales:

            if est_doublon(
                troupe,
                existant
            ):

                doublon = True

                print(
                    "Doublon ignoré -> "
                    f"{troupe.get('nom')} "
                    f"{troupe['type']} "
                    f"{troupe['nombre']}"
                )

                break

        if not doublon:

            troupes_finales.append(
                troupe
            )

            print(
                "Nouvelle ligne conservée -> "
                f"{troupe.get('nom')} "
                f"{troupe['type']} "
                f"{troupe['nombre']}"
            )

    # =====================================================
    # RECUPERATION D'UNE SEULE LIGNE ILISIBLE
    # =====================================================
    #
    # On ne se sert PLUS des blessés.
    #
    # Cette fonctionnalité est donc désactivée ici.
    #
    # Une ligne illisible reste simplement illisible.
    #

    if toutes_les_lignes_illisibles:

        print()

        print(
            f"⚠️ {len(toutes_les_lignes_illisibles)} "
            "ligne(s) n'ont pas pu être lues."
        )

    # =====================================================
    # TOTAL T4
    # =====================================================

    total_t4 = sum(
        troupe["nombre"]
        for troupe in troupes_finales
        if troupe["type"] == "T4"
    )

    # =====================================================
    # TOTAL T5
    # =====================================================

    total_t5 = sum(
        troupe["nombre"]
        for troupe in troupes_finales
        if troupe["type"] == "T5"
    )

    # =====================================================
    # TOTAL GENERAL
    # =====================================================

    total = (
        total_t4
        +
        total_t5
    )

    # =====================================================
    # RESSOURCES
    # =====================================================

    # =====================================================
    # RESSOURCES
    # =====================================================
    #
    # Lors d'une vérification avec deux captures, l'une peut être
    # en version PC et l'autre en version téléphone. On prend,
    # pour chaque ressource, la première valeur réellement détectée.
    # Ainsi une ressource absente sur l'une des captures peut être
    # récupérée depuis l'autre si elle y est visible.

    nourriture = None
    bois = None
    pierre = None
    or_ = None

    for image_result in resultats_images:

        if nourriture is None:
            nourriture = image_result["nourriture"]

        if bois is None:
            bois = image_result["bois"]

        if pierre is None:
            pierre = image_result["pierre"]

        if or_ is None:
            or_ = image_result["or"]

    # =====================================================
    # RESULTAT
    # =====================================================

    return {

        "t4":
            total_t4,

        "t5":
            total_t5,

        "total":
            total,

        "nourriture":
            nourriture,

        "bois":
            bois,

        "pierre":
            pierre,

        "or":
            or_
    }

if __name__ == "__main__":
    images = sys.argv[1:] or ["test_hopital1.png"]
    resultat = analyser_plusieurs_images(images)
    print()
    print("=" * 40)
    print("RESULTAT FINAL")
    print("=" * 40)
    print(f"T4            : {resultat['t4']}")
    print(f"T5            : {resultat['t5']}")
    print(f"Total troupes : {resultat['total']}")
    print(f"Nourriture    : {resultat['nourriture']}")
    print(f"Bois          : {resultat['bois']}")
    print(f"Pierre        : {resultat['pierre']}")
    print(f"Or            : {resultat['or']}")
