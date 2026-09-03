import cv2
import pytesseract
import numpy as np
import re
import sys
import unicodedata
from collections import Counter


# =========================================================
# CONFIGURATION
# =========================================================

# Tesseract :
# - sur Windows local, on utilise l'installation habituelle ;
# - sur Railway/Linux, shutil.which("tesseract") trouve le binaire
#   installé par le Dockerfile.
import os
import shutil

_tesseract_env = os.getenv("TESSERACT_CMD")

if _tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_env
else:
    _tesseract_linux = shutil.which("tesseract")

    if _tesseract_linux:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_linux
    else:
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )


# =========================================================
# LISTE DES UNITES T4
# =========================================================

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


# =========================================================
# LISTE DES UNITES T5
# =========================================================

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


# =========================================================
# NORMALISATION DES TEXTES
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


# =========================================================
# CREATION TABLEAU T4 / T5
# =========================================================

UNIT_TIERS = {}

for nom in T4_UNITS:

    UNIT_TIERS[
        normaliser_texte(nom)
    ] = "T4"


for nom in T5_UNITS:

    UNIT_TIERS[
        normaliser_texte(nom)
    ] = "T5"


# Noms normalisés triés du plus long au plus court.
# Le nom de l'unité, et non le nombre affiché à droite,
# détermine le T4/T5.
UNIT_NAMES_SORTED = sorted(
    UNIT_TIERS.keys(),
    key=len,
    reverse=True
)


# =========================================================
# PREPARATION D'UNE ZONE
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
# DETECTION DES LIGNES DE TROUPES
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
# OCR DU PANNEAU DES TROUPES
# =========================================================

def ocr_panneau_troupes(
    image
):

    h, w = image.shape[:2]

    # Zone contenant les noms et quantités
    x1 = int(
        w * 0.39
    )

    x2 = int(
        w * 0.94
    )

    y1 = int(
        h * 0.10
    )

    y2 = int(
        h * 0.72
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    crop = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    data = pytesseract.image_to_data(
        crop,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    mots = []

    for i, texte in enumerate(
        data["text"]
    ):

        texte = texte.strip()

        if not texte:

            continue

        try:

            confiance = float(
                data["conf"][i]
            )

        except ValueError:

            confiance = 0

        left = (
            data["left"][i]
            /
            3
            +
            x1
        )

        top = (
            data["top"][i]
            /
            3
            +
            y1
        )

        largeur = (
            data["width"][i]
            /
            3
        )

        hauteur = (
            data["height"][i]
            /
            3
        )

        centre_y = (
            top
            +
            hauteur / 2
        )

        mots.append(
            {
                "texte":
                    texte,

                "x":
                    left,

                "y":
                    centre_y,

                "confiance":
                    confiance
            }
        )

    return mots


# =========================================================
# DETECTION DES LIGNES PAR OCR (FALLBACK)
# =========================================================

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
# RECHERCHE DU NOM DE L'UNITE ET DU TIER
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


def trouver_tier(
    mots
):

    _, tier = trouver_unite(
        mots
    )

    return tier


# =========================================================
# RECHERCHE DU NOMBRE
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
# ANALYSE D'UNE LIGNE
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
# CONVERSION RESSOURCE
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
# DETECTION DE LA BARRE DE RESSOURCES
# =========================================================

def ressources_visibles(
    image
):
    """
    Détermine si la barre des ressources est réellement affichée.

    Quand les soins ont déjà été lancés, le jeu remplace la barre
    Food / Wood / Stone / Gold par une barre de temps avec un texte
    du type 00:42:51.

    On considère donc les ressources présentes seulement si :
    - le texte de la zone ne contient pas de ':'
    - au moins deux valeurs contiennent K/M/B

    Cette vérification évite notamment de prendre le temps restant
    pour une quantité de ressources.
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.40)
    x2 = int(w * 0.97)

    y1 = int(h * 0.68)
    y2 = int(h * 0.82)

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return False

    crop = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    textes = []

    textes.append(
        pytesseract.image_to_string(
            gray,
            config="--psm 7"
        )
    )

    for seuil in [
        120,
        140,
        160,
        180,
    ]:

        _, binary = cv2.threshold(
            gray,
            seuil,
            255,
            cv2.THRESH_BINARY
        )

        textes.append(
            pytesseract.image_to_string(
                binary,
                config="--psm 7"
            )
        )

    for texte in textes:

        texte_upper = texte.upper()

        # La présence d'un séparateur ':' indique la barre
        # de temps des soins, pas la barre de ressources.
        if ":" in texte_upper:
            continue

        matches = re.findall(
            r"\d+(?:[.,]\d+)?\s*[KMB]",
            texte_upper
        )

        if len(matches) >= 2:
            return True

    return False


# =========================================================
# LECTURE D'UNE RESSOURCE
# =========================================================

def lire_ressource(
    image,
    x1,
    y1,
    x2,
    y2
):

    crop = preparer_crop(
        image,
        x1,
        y1,
        x2,
        y2,
        scale=7
    )

    if crop is None:

        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    valeurs = []

    essais = [
        gray
    ]

    for seuil in [
        120,
        140,
        160,
        180
    ]:

        _, binary = cv2.threshold(
            gray,
            seuil,
            255,
            cv2.THRESH_BINARY
        )

        essais.append(
            binary
        )

    for image_ocr in essais:

        texte = pytesseract.image_to_string(
            image_ocr,
            config=(
                "--psm 7 "
                "-c tessedit_char_whitelist="
                "0123456789.KMB"
            )
        )

        valeur = convertir_ressource(
            texte
        )

        if valeur is not None:

            valeurs.append(
                valeur
            )

    if not valeurs:

        return None

    # Vote majoritaire
    compteur = Counter(
        valeurs
    )

    meilleur = compteur.most_common(
        1
    )[0][0]

    return meilleur


# =========================================================
# ANALYSE DES RESSOURCES
# =========================================================

def analyser_ressources(
    image
):

    # -----------------------------------------------------
    # Aucun soin lancé / ressources visibles
    # -----------------------------------------------------

    if not ressources_visibles(
        image
    ):

        print()
        print(
            "Ressources : non affichées "
            "(soins déjà lancés ou barre absente)."
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

    # -----------------------------------------------------
    # Ressources visibles -> lecture normale
    # -----------------------------------------------------

    nourriture = lire_ressource(
        image,
        0.41,
        0.72,
        0.55,
        0.81
    )

    bois = lire_ressource(
        image,
        0.54,
        0.72,
        0.68,
        0.81
    )

    pierre = lire_ressource(
        image,
        0.68,
        0.72,
        0.82,
        0.81
    )

    or_ = lire_ressource(
        image,
        0.82,
        0.72,
        0.96,
        0.81
    )

    print()
    print(
        "Nourriture :",
        nourriture
    )

    print(
        "Bois       :",
        bois
    )

    print(
        "Pierre     :",
        pierre
    )

    print(
        "Or         :",
        or_
    )

    return {

        "nourriture":
            nourriture,

        "bois":
            bois,

        "pierre":
            pierre,

        "or":
            or_
    }


# =========================================================
# ANALYSE D'UNE IMAGE
# =========================================================

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

    print()
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
# DOUBLON
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
# ANALYSE PLUSIEURS IMAGES
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

    premiere = resultats_images[0]

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
            premiere[
                "nourriture"
            ],

        "bois":
            premiere[
                "bois"
            ],

        "pierre":
            premiere[
                "pierre"
            ],

        "or":
            premiere[
                "or"
            ]
    }


# =========================================================
# PROGRAMME PRINCIPAL
# =========================================================

if __name__ == "__main__":

    images = sys.argv[1:]

    if not images:

        images = [
            "test_hopital1.png"
        ]

    resultat = analyser_plusieurs_images(
        images
    )

    print()
    print()

    print(
        "========================================"
    )

    print(
        "           RESULTAT FINAL"
    )

    print(
        "========================================"
    )

    print(
        f"T4             : "
        f"{resultat['t4']}"
    )

    print(
        f"T5             : "
        f"{resultat['t5']}"
    )

    print(
        f"Total troupes  : "
        f"{resultat['total']}"
    )

    print()

    print(
        f"Nourriture     : "
        f"{resultat['nourriture']}"
    )

    print(
        f"Bois           : "
        f"{resultat['bois']}"
    )

    print(
        f"Pierre         : "
        f"{resultat['pierre']}"
    )

    print(
        f"Or             : "
        f"{resultat['or']}"
    )
