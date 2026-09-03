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
# - Railway/Linux : utilise automatiquement le binaire installé dans PATH
# - Windows local : utilise l'installation standard
# - TESSERACT_CMD : permet de forcer un chemin si nécessaire
import os
import shutil

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
# RESSOURCES
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


def extraire_valeurs_barre_ressources(
    image
):
    """
    Lecture robuste de la barre Food/Wood/Stone/Gold sur PC et téléphone.

    Layout téléphone observé :
        2.1K  1.6K  144
        -> Wood / Stone / Gold

    Layout PC observé :
        33.1M  10.4M  17.7M  2.2M
        -> Food / Wood / Stone / Gold
    """

    h, w = image.shape[:2]

    # ---------------------------------------------------------
    # PASSAGE 1 : OCR de la barre entière
    # ---------------------------------------------------------
    #
    # La zone verticale est volontairement étroite pour exclure
    # le timer de soin situé juste en dessous.
    # ---------------------------------------------------------

    x1 = int(
        w * 0.25
    )

    x2 = int(
        w * 0.99
    )

    y1 = int(
        h * 0.70
    )

    y2 = int(
        h * 0.86
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return []

    candidats_sequences = []

    for scale in [
        4,
        6,
        8
    ]:

        agrandi = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        variantes = [
            agrandi,
            cv2.cvtColor(
                agrandi,
                cv2.COLOR_BGR2GRAY
            )
        ]

        for variante in variantes:

            texte = pytesseract.image_to_string(
                variante,
                config=(
                    "--psm 7 "
                    "-c tessedit_char_whitelist="
                    "0123456789.KMB,"
                )
            )

            texte = re.sub(
                r"\s+",
                " ",
                texte.upper()
            ).strip()

            if not texte:
                continue

            # -------------------------------------------------
            # On récupère les montants AVEC suffixe d'abord.
            # C'est le format normal pour Food/Wood/Stone.
            # -------------------------------------------------

            suffix_matches = list(
                re.finditer(
                    r"\d+(?:[.,]\d+)?\s*[KMB]",
                    texte
                )
            )

            sequence = []

            for match in suffix_matches:

                valeur = convertir_ressource(
                    match.group(0)
                )

                if valeur is not None:
                    sequence.append(
                        valeur
                    )

            # -------------------------------------------------
            # Gold peut être affiché sans suffixe :
            # téléphone -> 144.
            #
            # On cherche UNIQUEMENT un nombre non-suffixé situé
            # après le dernier montant suffixé, et proche de celui-ci.
            # Cela évite les faux nombres comme "5" ou "7" issus
            # du bouton/timer.
            # -------------------------------------------------

            if suffix_matches:

                fin_dernier_suffixe = (
                    suffix_matches[-1].end()
                )

                apres = texte[
                    fin_dernier_suffixe:
                ]

                # Premier nombre court après le dernier suffixe.
                # On limite à 3 chiffres pour le format Gold courant.
                match_gold = re.search(
                    r"^[\s,._-]*(\d{1,3})(?:\s|$)",
                    apres
                )

                if match_gold:

                    gold = int(
                        match_gold.group(1)
                    )

                    sequence.append(
                        gold
                    )

            # -------------------------------------------------
            # Cas où l'OCR a reconnu directement 3 ou 4 montants.
            # -------------------------------------------------

            if len(sequence) in (
                3,
                4
            ):

                candidats_sequences.append(
                    sequence
                )

    # ---------------------------------------------------------
    # Choix de la séquence la plus cohérente.
    # ---------------------------------------------------------

    if candidats_sequences:

        # On privilégie les séquences les plus longues et
        # les plus répétées par les différentes passes OCR.
        compte = {}

        for sequence in candidats_sequences:

            cle = tuple(
                sequence
            )

            compte[cle] = (
                compte.get(
                    cle,
                    0
                )
                +
                1
            )

        meilleure = max(
            compte.items(),
            key=lambda item: (
                item[1],
                len(item[0])
            )
        )[0]

        return [
            (
                index,
                valeur
            )
            for index, valeur in enumerate(
                meilleure
            )
        ]

    # ---------------------------------------------------------
    # PASSAGE 2 : fallback OCR dynamique
    # ---------------------------------------------------------

    x1 = int(
        w * 0.25
    )

    x2 = int(
        w * 0.99
    )

    y1 = int(
        h * 0.64
    )

    y2 = int(
        h * 0.88
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return []

    essais = []

    for scale in [
        4,
        6,
        8
    ]:

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

        essais.append(
            (
                gray,
                scale
            )
        )

        for seuil in [
            120,
            150,
            180
        ]:

            _, binary = cv2.threshold(
                gray,
                seuil,
                255,
                cv2.THRESH_BINARY
            )

            essais.append(
                (
                    binary,
                    scale
                )
            )

    detections = []

    for image_ocr, scale in essais:

        data = pytesseract.image_to_data(
            image_ocr,
            config=(
                "--psm 6 "
                "-c tessedit_char_whitelist="
                "0123456789.KMB,"
            ),
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
                confiance = 0

            if confiance < 15:
                continue

            propre = (
                brut.upper()
                .replace(",", ".")
                .replace(" ", "")
            )

            valeur = convertir_ressource(
                propre
            )

            if valeur is None:
                continue

            a_suffixe = propre.endswith(
                (
                    "K",
                    "M",
                    "B"
                )
            )

            # Les valeurs sans suffixe ne sont acceptées que
            # si elles sont plausibles pour Gold et dans la partie
            # droite de la barre.
            x = (
                data["left"][i] / scale
                +
                x1
            )

            if (
                not a_suffixe
                and
                not (
                    valeur <= 999
                    and
                    x / w >= 0.70
                )
            ):
                continue

            detections.append(
                (
                    x,
                    valeur
                )
            )

    if not detections:
        return []

    # Regrouper les valeurs identiques proches.
    detections = sorted(
        detections,
        key=lambda item:
        item[0]
    )

    groupes = []

    for x, valeur in detections:

        trouve = False

        for groupe in groupes:

            if (
                abs(
                    x
                    -
                    groupe["x"]
                )
                <=
                55
                and
                valeur
                ==
                groupe["valeur"]
            ):

                groupe["votes"] += 1
                groupe["xs"].append(
                    x
                )

                groupe["x"] = (
                    sum(
                        groupe["xs"]
                    )
                    /
                    len(
                        groupe["xs"]
                    )
                )

                trouve = True
                break

        if not trouve:

            groupes.append(
                {
                    "x":
                        x,
                    "xs":
                        [x],
                    "valeur":
                        valeur,
                    "votes":
                        1
                }
            )

    groupes = sorted(
        groupes,
        key=lambda groupe:
        groupe["x"]
    )

    # Ne conserver que des valeurs soutenues.
    groupes_soutenus = [
        groupe
        for groupe in groupes
        if groupe["votes"] >= 2
    ]

    if groupes_soutenus:

        groupes = groupes_soutenus

    return [
        (
            groupe["x"],
            groupe["valeur"]
        )
        for groupe in groupes
    ]


def ressources_visibles(
    image
):
    """
    Une barre de ressources est considérée présente si l'OCR
    trouve au moins deux montants cohérents dans sa zone.

    Le timer de soins (00:42:51, etc.) ne passe pas ce filtre.
    """

    valeurs = extraire_valeurs_barre_ressources(
        image
    )

    return len(valeurs) >= 2


def lire_ressource(
    image,
    x1,
    y1,
    x2,
    y2
):
    """
    Fonction conservée pour compatibilité.
    """

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

    compteur = Counter(
        valeurs
    )

    return compteur.most_common(
        1
    )[0][0]


# =========================================================
# ANALYSE DES RESSOURCES
# =========================================================


def _classer_icon_ressource(
    image,
    x,
    y,
    largeur,
    hauteur
):
    """
    Classe l'icône placée à gauche du montant OCR.

    Important :
    Le token OCR de la ressource peut inclure une partie de l'icône.
    On analyse donc la partie gauche du bounding-box OCR au lieu de
    chercher l'icône à une distance fixe.

    Retour :
        ("food" | "wood" | "stone" | "gold" | None, score)
    """

    h, w = image.shape[:2]

    # La partie gauche du token contient généralement l'icône.
    px1 = max(
        0,
        int(x)
    )

    px2 = min(
        w,
        int(
            x
            +
            max(
                18,
                largeur * 0.45
            )
        )
    )

    py1 = max(
        0,
        int(
            y - max(
                5,
                hauteur * 0.15
            )
        )
    )

    py2 = min(
        h,
        int(
            y
            +
            hauteur
            +
            max(
                5,
                hauteur * 0.15
            )
        )
    )

    patch = image[
        py1:py2,
        px1:px2
    ]

    if patch.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2HSV
    )

    H, S, V = cv2.split(
        hsv
    )

    # BGR pour détecter proprement les éléments gris.
    B, G, R = cv2.split(
        patch
    )

    green = int(
        (
            (H >= 35)
            &
            (H < 95)
            &
            (S > 90)
            &
            (V > 50)
        ).sum()
    )

    orange = int(
        (
            (H < 25)
            &
            (S > 90)
            &
            (V > 50)
        ).sum()
    )

    yellow = int(
        (
            (H >= 20)
            &
            (H < 48)
            &
            (S > 100)
            &
            (V > 70)
        ).sum()
    )

    # Pixels vraiment gris/neutres.
    grey = int(
        (
            (np.maximum.reduce(
                [B, G, R]
            )
            -
            np.minimum.reduce(
                [B, G, R]
            )
            <
            35)
            &
            (V > 100)
        ).sum()
    )

    # Pierre peut aussi avoir une teinte bleue/grise.
    blue_grey = int(
        (
            (H >= 85)
            &
            (H <= 135)
            &
            (S > 20)
            &
            (S < 150)
            &
            (V > 100)
        ).sum()
    )

    total = max(
        patch.shape[0] * patch.shape[1],
        1
    )

    scores = {
        "food":
            green / total,

        "wood":
            orange / total,

        "stone":
            (
                grey
                +
                blue_grey * 0.75
            )
            /
            total,

        "gold":
            yellow / total
    }

    # ---------------------------------------------------------
    # FOOD
    # ---------------------------------------------------------
    #
    # Le maïs comporte une forte composante verte + jaune.
    # Le vert permet de le distinguer de l'or.
    if (
        green >= 20
        and
        green >= orange * 1.35
        and
        green >= yellow * 0.70
    ):
        return (
            "food",
            scores["food"]
        )

    # ---------------------------------------------------------
    # WOOD
    # ---------------------------------------------------------
    #
    # Le rondin est très fortement orange.
    if (
        orange >= 40
        and
        orange >= green * 1.5
        and
        orange >= yellow * 1.15
    ):
        return (
            "wood",
            scores["wood"]
        )

    # ---------------------------------------------------------
    # GOLD
    # ---------------------------------------------------------
    #
    # La pièce est jaune/orange, mais beaucoup moins orange pur
    # qu'un rondin et quasiment sans vert.
    if (
        yellow >= 20
        and
        yellow >= green * 1.5
        and
        yellow >= orange * 0.75
    ):
        return (
            "gold",
            scores["gold"]
        )

    # ---------------------------------------------------------
    # STONE
    # ---------------------------------------------------------
    if (
        grey >= 30
        and
        grey + blue_grey >= green * 1.2
        and
        grey + blue_grey >= orange * 0.8
    ):
        return (
            "stone",
            scores["stone"]
        )

    return None, 0.0


def _lire_ressources_avec_icones(
    image
):
    """
    Trouve la ligne des ressources par OCR puis identifie chaque
    montant avec l'icône qui lui est associée.

    Contrairement à l'ancienne logique, on ne suppose jamais que
    3 montants signifient une combinaison précise.

    Exemples acceptés :
        Food / Wood / Gold
        Wood / Stone / Gold
        Food / Wood / Stone
        Food / Wood / Stone / Gold
    """

    h, w = image.shape[:2]

    # On élimine la barre de ressources globale du jeu située tout en haut.
    crop_y1 = int(
        h * 0.55
    )

    crop_y2 = int(
        h * 0.94
    )

    crop = image[
        crop_y1:crop_y2,
        :
    ]

    if crop.size == 0:
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

    candidats = []

    # Plusieurs résolutions/PSM pour améliorer la détection du montant.
    for scale in (
        3,
        4,
        5,
        6
    ):

        agrandi = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        variantes = [
            agrandi,
            cv2.cvtColor(
                agrandi,
                cv2.COLOR_BGR2GRAY
            )
        ]

        for variante in variantes:

            for psm in (
                6,
                11
            ):

                data = pytesseract.image_to_data(
                    variante,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist="
                        "0123456789.KMB,"
                    ),
                    output_type=pytesseract.Output.DICT
                )

                for i, brut in enumerate(
                    data["text"]
                ):

                    brut = brut.strip()

                    if not brut:
                        continue

                    propre = (
                        brut
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

                    # Montants autorisés :
                    # 2.0K / 144 / 33.1M / 2B...
                    if not re.fullmatch(
                        r"\d+(?:\.\d+)?[KMB]?",
                        propre
                    ):
                        continue

                    valeur = convertir_ressource(
                        propre
                    )

                    if valeur is None:
                        continue

                    x = (
                        data["left"][i]
                        /
                        scale
                    )

                    y = (
                        data["top"][i]
                        /
                        scale
                        +
                        crop_y1
                    )

                    largeur = (
                        data["width"][i]
                        /
                        scale
                    )

                    hauteur = (
                        data["height"][i]
                        /
                        scale
                    )

                    # Trop petits nombres du décor / boutons.
                    if (
                        largeur < 10
                        or
                        hauteur < 8
                    ):
                        continue

                    resource, score = (
                        _classer_icon_ressource(
                            image,
                            x,
                            y,
                            largeur,
                            hauteur
                        )
                    )

                    if resource is None:
                        continue

                    # Food/Wood/Stone should normally have K/M/B.
                    # A bare number such as "2" is often an OCR fragment
                    # of "2.0K". Gold is the exception (e.g. 137/144).
                    if (
                        not propre.endswith(
                            (
                                "K",
                                "M",
                                "B"
                            )
                        )
                        and
                        resource != "gold"
                    ):
                        continue

                    candidats.append(
                        {
                            "resource":
                                resource,

                            "value":
                                valeur,

                            "x":
                                x,

                            "y":
                                y,

                            "score":
                                score
                        }
                    )

    if not candidats:
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

    # Eliminate numbers from the left side of the hospital panel
    # (wounded totals/capacities). The resource row is on the right.
    candidats = [
        candidat
        for candidat in candidats
        if candidat["x"] >= w * 0.45
        and candidat["y"] >= h * 0.55
        and candidat["y"] <= h * 0.92
    ]

    if not candidats:
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

    # ---------------------------------------------------------
    # Garder uniquement la vraie ligne de ressources.
    #
    # Il existe d'autres nombres dans le panneau (19/720 000,
    # 0/50 000, timer, etc.). Les ressources sont regroupées sur
    # une même ligne horizontale. On sélectionne donc le groupe
    # de détections qui possède au moins 2 ressources et la meilleure
    # qualité globale.
    # ---------------------------------------------------------

    groupes_y = []

    for candidat in candidats:

        centre_y = (
            candidat["y"]
            +
            0.5
        )

        groupe_trouve = None

        for groupe in groupes_y:

            if (
                abs(
                    centre_y
                    -
                    groupe["y"]
                )
                <=
                35
            ):

                groupe["items"].append(
                    candidat
                )

                groupe["y"] = (
                    sum(
                        item["y"]
                        for item in groupe["items"]
                    )
                    /
                    len(
                        groupe["items"]
                    )
                )

                groupe_trouve = groupe

                break

        if groupe_trouve is None:

            groupes_y.append(
                {
                    "y":
                        candidat["y"],

                    "items":
                        [candidat]
                }
            )

    # Pour chaque groupe, ne conserver qu'une lecture par ressource/x.
    groupes_valides = []

    for groupe in groupes_y:

        # Déduplication spatiale.
        uniques = []

        for item in sorted(
            groupe["items"],
            key=lambda element:
            element["x"]
        ):

            proche = False

            for precedent in uniques:

                if (
                    abs(
                        item["x"]
                        -
                        precedent["x"]
                    )
                    <=
                    45
                    and
                    item["resource"]
                    ==
                    precedent["resource"]
                ):

                    proche = True

                    if item["score"] > precedent["score"]:
                        uniques.remove(precedent)
                        uniques.append(item)

                    break

            if not proche:
                uniques.append(item)

        ressources_groupe = {
            item["resource"]
            for item in uniques
        }

        if len(
            ressources_groupe
        ) >= 2:

            score_groupe = (
                len(ressources_groupe)
                * 100
                +
                sum(
                    item["score"]
                    for item in uniques
                )
            )

            groupes_valides.append(
                {
                    "items":
                        uniques,

                    "score":
                        score_groupe
                }
            )

    if groupes_valides:

        meilleur_groupe = max(
            groupes_valides,
            key=lambda groupe:
            groupe["score"]
        )

        candidats = meilleur_groupe[
            "items"
        ]

    # ---------------------------------------------------------
    # Fusionner les morceaux OCR d'un même montant.
    #
    # Exemple téléphone :
    #   2.0K
    #
    # Tesseract peut aussi produire :
    #   2
    #   0K
    #
    # Les deux détections ont presque le même X. On garde celle qui
    # correspond le mieux au type d'icône. Cela évite de transformer
    # un seul montant Wood en Wood + Stone.
    # ---------------------------------------------------------

    candidats = sorted(
        candidats,
        key=lambda item:
        item["x"]
    )

    groupes_x = []

    for candidat in candidats:

        ajoute = False

        for groupe in groupes_x:

            if (
                abs(
                    candidat["x"]
                    -
                    groupe["x"]
                )
                <=
                65
            ):

                groupe["items"].append(
                    candidat
                )

                groupe["x"] = (
                    sum(
                        item["x"]
                        for item in groupe["items"]
                    )
                    /
                    len(
                        groupe["items"]
                    )
                )

                ajoute = True
                break

        if not ajoute:

            groupes_x.append(
                {
                    "x":
                        candidat["x"],

                    "items":
                        [candidat]
                }
            )

    candidats_nettoyes = []

    for groupe in groupes_x:

        # Voter d'abord pour le type d'icône.
        par_type = {}

        for item in groupe["items"]:

            resource = item["resource"]

            par_type.setdefault(
                resource,
                []
            ).append(
                item
            )

        resource, items = max(
            par_type.items(),
            key=lambda element: (
                len(
                    element[1]
                ),
                sum(
                    item["score"]
                    for item in element[1]
                )
            )
        )

        # Puis voter pour la valeur OCR.
        valeurs = {}

        for item in items:

            valeur = item["value"]

            valeurs.setdefault(
                valeur,
                {
                    "votes":
                        0,

                    "score":
                        0.0
                }
            )

            valeurs[
                valeur
            ]["votes"] += 1

            valeurs[
                valeur
            ]["score"] += item["score"]

        valeur, _ = max(
            valeurs.items(),
            key=lambda element: (
                element[1]["votes"],
                element[1]["score"]
            )
        )

        candidats_nettoyes.append(
            {
                "resource":
                    resource,

                "value":
                    valeur,

                "x":
                    groupe["x"],

                "score":
                    sum(
                        item["score"]
                        for item in items
                    )
            }
        )

    candidats = candidats_nettoyes

    # ---------------------------------------------------------
    # Regrouper les lectures restantes et voter par ressource.
    # ---------------------------------------------------------

    par_ressource = {
        "food": [],
        "wood": [],
        "stone": [],
        "gold": []
    }

    for candidat in candidats:

        par_ressource[
            candidat["resource"]
        ].append(
            candidat
        )

    resultats = {}

    for resource, items in par_ressource.items():

        if not items:

            resultats[
                resource
            ] = None

            continue

        votes = {}

        for item in items:

            cle = (
                item["value"]
            )

            if cle not in votes:

                votes[cle] = {
                    "count":
                        0,

                    "score":
                        0.0
                }

            votes[cle]["count"] += 1

            votes[cle]["score"] += (
                item["score"]
            )

        meilleur = max(
            votes.items(),
            key=lambda item: (
                item[1]["count"],
                item[1]["score"]
            )
        )

        resultats[
            resource
        ] = (
            meilleur[0]
            if meilleur[0] > 0
            else None
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



def analyser_ressources(
    image
):

    detections = (
        extraire_valeurs_barre_ressources(
            image
        )
    )

    if len(detections) < 2:

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

    # ---------------------------------------------------------
    # Les ressources sont dans l'ordre gauche -> droite.
    #
    # 4 valeurs :
    # Food / Wood / Stone / Gold
    #
    # 3 valeurs :
    # Dans la version téléphone montrée, Food n'est pas affichée
    # et on a Wood / Stone / Gold.
    # ---------------------------------------------------------

    valeurs = [
        valeur
        for _, valeur in detections
    ]

    # ---------------------------------------------------------
    # IDENTIFICATION DU TYPE DE RESSOURCE PAR L'ICÔNE
    # ---------------------------------------------------------
    #
    # Ne plus supposer :
    #   3 valeurs = Wood / Stone / Gold
    #
    # Une capture téléphone peut par exemple contenir :
    #   Food / Wood / Gold
    #
    # Les icônes sont donc détectées visuellement et associées
    # aux valeurs dans leur ordre gauche -> droite.
    # ---------------------------------------------------------

    nourriture = None
    bois = None
    pierre = None
    or_ = None

    icones = _trouver_icones_ressources(
        image
    )

    if len(icones) == len(valeurs):

        for icon, valeur in zip(
            icones,
            valeurs
        ):

            if icon["resource"] == "food":

                nourriture = valeur

            elif icon["resource"] == "wood":

                bois = valeur

            elif icon["resource"] == "stone":

                pierre = valeur

            elif icon["resource"] == "gold":

                or_ = valeur

    elif len(valeurs) >= 4:

        # Fallback PC historique si une icône est trop faible
        # pour être détectée.
        nourriture = valeurs[0]
        bois = valeurs[1]
        pierre = valeurs[2]
        or_ = valeurs[3]

    elif len(valeurs) == 3:

        # Fallback téléphone historique.
        nourriture = None
        bois = valeurs[0]
        pierre = valeurs[1]
        or_ = valeurs[2]

    else:

        # Cas inhabituel : 2 ressources visibles.
        # On utilise leurs positions horizontales pour déterminer
        # les ressources connues.
        positions = [
            x
            for x, _ in detections
        ]

        nourriture = None
        bois = None
        pierre = None
        or_ = None

        # Position relative dans la barre.
        for x, valeur in detections:

            relative = x / image.shape[1]

            if relative < 0.55:

                # Dans la version téléphone observée, le premier
                # montant visible à cet endroit est le bois.
                bois = valeur

            elif relative < 0.78:

                pierre = valeur

            else:

                or_ = valeur

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
