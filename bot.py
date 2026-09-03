import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime

import discord
from discord.ext import commands
import gspread
from google.oauth2.service_account import Credentials

from analyzer import analyser_plusieurs_images


# =========================================================
# CONFIGURATION
# =========================================================

TOKEN = os.getenv("DISCORD_TOKEN", "TON_TOKEN_ICI")

VERIFICATION_CHANNEL_ID = 1544613786952925326
RESULT_CHANNEL_ID = 1544613786952925326

# Les commandes de gestion seront utilisables uniquement
# dans le salon des résultats.
COMMANDS_CHANNEL_ID = RESULT_CHANNEL_ID

# Pour le moment, le bot conserve les messages et captures
# dans le salon de vérification, même après traitement ou erreur.
# Remettre cette valeur à True plus tard si on veut réactiver
# la suppression automatique.
DELETE_SOURCE_MESSAGES = False

# Nom du rôle Discord autorisé à utiliser !delete et !update.
# Les membres avec la permission Administrateur sont aussi autorisés.
ADMIN_ROLE_NAME = "Server • Admin"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

MAX_IMAGES = 2


# =========================================================
# GOOGLE SHEETS
# =========================================================

GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

GOOGLE_SHEET_ID = (
    "14fGzNfetRhUJCZYay8-woEr7heVLs3RuqiRAd-7EM34"
)

GOOGLE_WORKSHEET_NAME = "hopital"

EXPECTED_HEADERS = [
    "Date",
    "Player ID",
    "T4",
    "T5",
    "Total",
    "Food",
    "Wood",
    "Stone",
    "Gold",
]

UPDATE_FIELDS = {
    "t4": "T4",
    "t5": "T5",
    "total": "Total",
    "food": "Food",
    "wood": "Wood",
    "stone": "Stone",
    "gold": "Gold",
}


def obtenir_feuille_google():
    """
    Connexion au compte de service et ouverture de l'onglet.
    """

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    google_credentials_json = os.getenv(
        "GOOGLE_CREDENTIALS_JSON"
    )

    if google_credentials_json:

        try:
            credentials_info = json.loads(
                google_credentials_json
            )

        except json.JSONDecodeError as e:

            raise ValueError(
                "GOOGLE_CREDENTIALS_JSON n'est pas un JSON valide."
            ) from e

        credentials = Credentials.from_service_account_info(
            credentials_info,
            scopes=scopes,
        )

    else:

        credentials = Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_FILE,
            scopes=scopes,
        )

    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(
        GOOGLE_SHEET_ID
    )

    try:
        worksheet = spreadsheet.worksheet(
            GOOGLE_WORKSHEET_NAME
        )
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.sheet1

    return worksheet


def verifier_entetes(worksheet):
    """
    Vérifie la première ligne du Sheet.
    Si elle est vide, crée les en-têtes.
    """

    premiere_ligne = worksheet.row_values(1)

    if not premiere_ligne:
        worksheet.update(
            "A1:I1",
            [EXPECTED_HEADERS],
        )
        return

    if premiere_ligne[:9] != EXPECTED_HEADERS:
        print(
            "⚠️ Attention : les en-têtes du Google Sheet "
            "ne correspondent pas exactement à :"
        )
        print(
            EXPECTED_HEADERS
        )


def ajouter_verification_google(
    player_id,
    t4,
    t5,
    total,
    nourriture,
    bois,
    pierre,
    or_,
):
    """
    Ajoute une nouvelle ligne dans l'historique.

    Colonnes :
    Date | Player ID | T4 | T5 | Total | Food | Wood | Stone | Gold
    """

    worksheet = obtenir_feuille_google()

    verifier_entetes(
        worksheet
    )

    date_heure = datetime.now().strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    ligne = [
        date_heure,
        player_id,
        t4,
        t5,
        total,
        nourriture if nourriture is not None else "",
        bois if bois is not None else "",
        pierre if pierre is not None else "",
        or_ if or_ is not None else "",
    ]

    worksheet.append_row(
        ligne,
        value_input_option="USER_ENTERED",
    )

    print(
        "✅ Vérification ajoutée dans Google Sheets."
    )


def trouver_lignes_player_id(
    worksheet,
    player_id,
):
    """
    Retourne les numéros de lignes du Sheet correspondant
    au Player ID demandé.
    """

    valeurs = worksheet.get_all_values()

    if not valeurs:
        return []

    # Première ligne = en-têtes.
    headers = valeurs[0]

    try:
        player_col = headers.index(
            "Player ID"
        )
    except ValueError:
        raise ValueError(
            "La colonne 'Player ID' est introuvable dans le Sheet."
        )

    lignes = []

    for row_number, row in enumerate(
        valeurs[1:],
        start=2,
    ):

        if player_col >= len(row):
            continue

        if str(row[player_col]).strip() == str(player_id).strip():

            lignes.append(
                row_number
            )

    return lignes


def player_id_existe_google(
    player_id,
):
    """
    Vérifie si le Player ID existe déjà dans l'onglet hopital.

    Retourne True dès qu'au moins une ligne correspond.
    Après suppression de toutes les lignes d'un joueur, le même
    Player ID pourra de nouveau être envoyé.
    """

    worksheet = obtenir_feuille_google()

    lignes = trouver_lignes_player_id(
        worksheet,
        player_id,
    )

    return len(lignes) > 0


def supprimer_player_id_google(
    player_id,
):
    """
    Supprime la ligne correspondant au Player ID.

    Les doublons étant bloqués, un Player ID ne doit normalement
    avoir qu'une seule ligne dans "hopital".
    """

    worksheet = obtenir_feuille_google()

    lignes = trouver_lignes_player_id(
        worksheet,
        player_id,
    )

    if not lignes:
        return False

    row_number = lignes[0]

    worksheet.delete_rows(
        row_number
    )

    return True


def colonne_google(
    worksheet,
    nom_colonne,
):
    """
    Retourne le numéro de colonne 1-based correspondant au nom.
    """

    headers = worksheet.row_values(1)

    try:
        return headers.index(
            nom_colonne
        ) + 1

    except ValueError:
        raise ValueError(
            f"La colonne '{nom_colonne}' "
            "est introuvable dans le Sheet."
        )


def convertir_entier_commande(
    texte,
):
    """
    Accepte :
    15000
    15 000
    15,000
    15.000
    """

    brut = (
        str(texte)
        .replace(" ", "")
        .replace(",", "")
        .replace(".", "")
    )

    if not brut.isdigit():
        raise ValueError(
            "La valeur doit être un nombre entier."
        )

    return int(
        brut
    )


def lire_ligne_google(
    worksheet,
    row_number,
):
    """
    Retourne un dictionnaire des valeurs d'une ligne.
    """

    headers = worksheet.row_values(1)
    values = worksheet.row_values(
        row_number
    )

    resultat = {}

    for index, header in enumerate(
        headers
    ):

        if index < len(values):
            resultat[header] = values[index]
        else:
            resultat[header] = ""

    return resultat


def modifier_derniere_verification_google(
    player_id,
    champ,
    valeur,
):
    """
    Modifie la dernière vérification correspondant au Player ID.

    Si T4 ou T5 est modifié, Total est recalculé automatiquement.
    """

    worksheet = obtenir_feuille_google()

    lignes = trouver_lignes_player_id(
        worksheet,
        player_id,
    )

    if not lignes:
        return None, None, None, None

    row_number = lignes[-1]

    champ_normalise = champ.strip().lower()

    if champ_normalise not in UPDATE_FIELDS:
        raise ValueError(
            "Stat invalide."
        )

    nom_colonne = UPDATE_FIELDS[
        champ_normalise
    ]

    nouvelle_valeur = convertir_entier_commande(
        valeur
    )

    col_index = colonne_google(
        worksheet,
        nom_colonne,
    )

    worksheet.update_cell(
        row_number,
        col_index,
        nouvelle_valeur,
    )

    # Si T4 ou T5 change, on recalcule Total.
    if champ_normalise in {
        "t4",
        "t5",
    }:

        ligne = lire_ligne_google(
            worksheet,
            row_number,
        )

        try:
            t4_actuel = convertir_entier_commande(
                ligne.get("T4", "0")
            )
        except ValueError:
            t4_actuel = 0

        try:
            t5_actuel = convertir_entier_commande(
                ligne.get("T5", "0")
            )
        except ValueError:
            t5_actuel = 0

        total = (
            t4_actuel
            +
            t5_actuel
        )

        total_col = colonne_google(
            worksheet,
            "Total",
        )

        worksheet.update_cell(
            row_number,
            total_col,
            total,
        )

    ligne_finale = lire_ligne_google(
        worksheet,
        row_number,
    )

    return (
        row_number,
        nom_colonne,
        nouvelle_valeur,
        ligne_finale,
    )


def historique_player_google(
    player_id,
    limite=10,
):
    """
    Retourne les dernières vérifications du joueur avec
    le vrai numéro de ligne Google Sheets.
    """

    worksheet = obtenir_feuille_google()

    lignes = trouver_lignes_player_id(
        worksheet,
        player_id,
    )

    if not lignes:
        return []

    lignes = lignes[-limite:]

    resultat = []

    for row_number in lignes:

        resultat.append(
            {
                "sheet_row":
                    row_number,

                "data":
                    lire_ligne_google(
                        worksheet,
                        row_number,
                    ),
            }
        )

    return resultat


def derniere_verification_google(
    player_id,
):
    historique = historique_player_google(
        player_id,
        limite=1,
    )

    if not historique:
        return None

    return historique[0]


# =========================================================
# STATISTIQUES DU GOOGLE SHEET
# =========================================================

def obtenir_statistiques_globales_google():
    """
    Lit l'onglet 'hopital' et calcule :

    - nombre de personnes actuellement enregistrées ;
    - total T4 ;
    - total T5 ;
    - puissance totale perdue.

    Formule de puissance :
    T4 x 4 + T5 x 10

    Le nombre de personnes est basé sur les Player IDs présents
    dans le Sheet. On compte les IDs uniques par sécurité.
    """

    worksheet = obtenir_feuille_google()

    valeurs = worksheet.get_all_values()

    if not valeurs:
        return {
            "joueurs":
                0,

            "t4":
                0,

            "t5":
                0,

            "puissance":
                0
        }

    headers = valeurs[0]

    try:
        player_col = headers.index(
            "Player ID"
        )

        t4_col = headers.index(
            "T4"
        )

        t5_col = headers.index(
            "T5"
        )

    except ValueError as e:

        raise ValueError(
            "Les colonnes Player ID, T4 ou T5 "
            "sont introuvables dans l'onglet hopital."
        ) from e

    player_ids = set()

    total_t4 = 0
    total_t5 = 0

    for row in valeurs[1:]:

        # Player ID
        if player_col < len(row):

            player_id = str(
                row[player_col]
            ).strip()

            if player_id:
                player_ids.add(
                    player_id
                )

        # T4
        if t4_col < len(row):

            brut_t4 = (
                str(row[t4_col])
                .replace(" ", "")
                .replace(",", "")
                .replace(".", "")
            )

            if brut_t4.isdigit():

                total_t4 += int(
                    brut_t4
                )

        # T5
        if t5_col < len(row):

            brut_t5 = (
                str(row[t5_col])
                .replace(" ", "")
                .replace(",", "")
                .replace(".", "")
            )

            if brut_t5.isdigit():

                total_t5 += int(
                    brut_t5
                )

    puissance = (
        total_t4 * 4
        +
        total_t5 * 10
    )

    return {
        "joueurs":
            len(player_ids),

        "t4":
            total_t4,

        "t5":
            total_t5,

        "puissance":
            puissance
    }


def formater_nombre(
    nombre
):
    return f"{nombre:,}".replace(
        ",",
        " "
    )


# =========================================================
# DISCORD
# =========================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None,
)

# Empêche deux vérifications d'être traitées simultanément.
# Ainsi, deux personnes ne peuvent pas envoyer le même Player ID
# au même moment et passer toutes les deux le contrôle de doublon
# avant l'écriture dans Google Sheets.
verification_lock = asyncio.Lock()


# =========================================================
# VERIFICATION DES COMMANDES
# =========================================================

def commande_dans_bon_salon():
    async def predicate(
        ctx,
    ):

        salons_autorises = {
            VERIFICATION_CHANNEL_ID,
            RESULT_CHANNEL_ID,
        }

        if ctx.channel.id not in salons_autorises:

            raise commands.CheckFailure(
                "Cette commande doit être utilisée "
                "dans le salon de vérification ou "
                "dans le salon des résultats."
            )

        return True

    return commands.check(
        predicate
    )


def admin_ou_role_autorise():
    async def predicate(
        ctx,
    ):

        if ctx.author.guild_permissions.administrator:
            return True

        for role in ctx.author.roles:

            if role.name == ADMIN_ROLE_NAME:
                return True

        raise commands.CheckFailure(
            "Tu n'as pas la permission d'utiliser "
            "cette commande."
        )

    return commands.check(
        predicate
    )


# =========================================================
# NOTIFICATION PRIVEE EN CAS D'ERREUR
# =========================================================

async def envoyer_dm_erreur(
    user,
    titre,
    description,
):
    """
    Essaie d'envoyer un message privé à l'utilisateur.

    Si les MP sont fermés, cela ne fait pas échouer la vérification.
    L'erreur principale reste visible dans le salon de résultats.
    """

    try:

        await user.send(
            f"❌ **ROK Hospital Checker — {titre}**\n\n"
            f"{description}"
        )

        print(
            f"📩 DM envoyé à {user}."
        )

        return True

    except discord.Forbidden:

        print(
            f"⚠️ Impossible d'envoyer un DM à {user} "
            "(MP fermés ou non autorisés)."
        )

        return False

    except discord.HTTPException as e:

        print(
            f"⚠️ Erreur Discord lors du DM à {user}: {e}"
        )

        return False


# =========================================================
# OUTILS DISCORD
# =========================================================

async def obtenir_salon_resultats():

    channel = bot.get_channel(
        RESULT_CHANNEL_ID
    )

    if channel is not None:
        return channel

    try:

        channel = await bot.fetch_channel(
            RESULT_CHANNEL_ID
        )

        return channel

    except Exception as e:

        print(
            "Impossible de récupérer le salon "
            f"de résultats : {e}"
        )

        return None


async def envoyer_images_resultat(
    image_paths,
):
    fichiers = []

    for image_path in image_paths:

        try:

            fichiers.append(
                discord.File(
                    image_path,
                    filename=os.path.basename(
                        image_path
                    ),
                )
            )

        except Exception as e:

            print(
                f"Impossible de préparer l'image "
                f"{image_path}: {e}"
            )

    return fichiers


# =========================================================
# EVENEMENT READY
# =========================================================

@bot.event
async def on_ready():

    print(
        "========================================"
    )

    print(
        "Emi's slave 2.0 connecté"
    )

    print(
        f"Compte : {bot.user}"
    )

    print(
        f"Salon vérification : "
        f"{VERIFICATION_CHANNEL_ID}"
    )

    print(
        f"Salon résultats    : "
        f"{RESULT_CHANNEL_ID}"
    )

    print(
        f"Google Sheet       : "
        f"{GOOGLE_SHEET_ID}"
    )

    print(
        f"Rôle admin         : "
        f"{ADMIN_ROLE_NAME}"
    )

    print(
        "========================================"
    )


# =========================================================
# COMMANDE !LATEST
# =========================================================

@bot.command(
    name="latest",
)
@commande_dans_bon_salon()
async def latest_command(
    ctx,
    player_id: str,
):

    if not player_id.isdigit():

        await ctx.send(
            "❌ Player ID invalide. "
            "Exemple : `!latest 219502046`"
        )

        return

    try:

        resultat = await asyncio.to_thread(
            derniere_verification_google,
            player_id,
        )

    except Exception as e:

        print(
            f"❌ Erreur !latest : {repr(e)}"
        )

        await ctx.send(
            "❌ Impossible de lire Google Sheets."
        )

        return

    if resultat is None:

        await ctx.send(
            f"❌ Aucune vérification trouvée "
            f"pour **{player_id}**."
        )

        return

    data = resultat["data"]
    sheet_row = resultat["sheet_row"]

    message = (
        "📊 **Latest verification**\n\n"
        f"👤 **Player ID:** {player_id}\n"
        f"📄 **Sheet row:** {sheet_row}\n"
        f"🕒 **Date:** {data.get('Date', '')}\n\n"
        f"🟪 **T4:** {data.get('T4', '')}\n"
        f"🟧 **T5:** {data.get('T5', '')}\n"
        f"⚔️ **Total:** {data.get('Total', '')}\n\n"
        f"🌾 **Food:** {data.get('Food', '')}\n"
        f"🪵 **Wood:** {data.get('Wood', '')}\n"
        f"🪨 **Stone:** {data.get('Stone', '')}\n"
        f"🪙 **Gold:** {data.get('Gold', '')}"
    )

    await ctx.send(
        message
    )


# =========================================================
# COMMANDE !HISTORY
# =========================================================

@bot.command(
    name="history",
)
@commande_dans_bon_salon()
async def history_command(
    ctx,
    player_id: str,
):

    if not player_id.isdigit():

        await ctx.send(
            "❌ Player ID invalide. "
            "Exemple : `!history 219502046`"
        )

        return

    try:

        historique = await asyncio.to_thread(
            historique_player_google,
            player_id,
            10,
        )

    except Exception as e:

        print(
            f"❌ Erreur !history : {repr(e)}"
        )

        await ctx.send(
            "❌ Impossible de lire Google Sheets."
        )

        return

    if not historique:

        await ctx.send(
            f"❌ Aucun historique trouvé "
            f"pour **{player_id}**."
        )

        return

    lignes = [
        "📚 **History - "
        f"{player_id}**",
        "",
    ]

    for index, entree in enumerate(
        historique,
        start=1,
    ):

        ligne = entree["data"]
        sheet_row = entree["sheet_row"]

        lignes.append(
            f"**#{index} — Sheet row {sheet_row} — "
            f"{ligne.get('Date', '')}**"
        )

        lignes.append(
            f"T4: **{ligne.get('T4', '')}** | "
            f"T5: **{ligne.get('T5', '')}** | "
            f"Total: **{ligne.get('Total', '')}**"
        )

        lignes.append(
            f"Food: {ligne.get('Food', '')} | "
            f"Wood: {ligne.get('Wood', '')} | "
            f"Stone: {ligne.get('Stone', '')} | "
            f"Gold: {ligne.get('Gold', '')}"
        )

        lignes.append("")

    texte = "\n".join(
        lignes
    )

    # Limite Discord : 2000 caractères.
    if len(texte) <= 2000:

        await ctx.send(
            texte
        )

    else:

        partie = texte[:1950] + "\n..."

        await ctx.send(
            partie
        )


# =========================================================
# COMMANDE !UPDATE
# =========================================================

@bot.command(
    name="update",
)
@commande_dans_bon_salon()
@admin_ou_role_autorise()
async def update_command(
    ctx,
    player_id: str,
    champ: str,
    valeur: str,
):

    if not player_id.isdigit():

        await ctx.send(
            "❌ Player ID invalide."
        )

        return

    champ_normalise = champ.strip().lower()

    if champ_normalise not in UPDATE_FIELDS:

        champs = ", ".join(
            UPDATE_FIELDS.keys()
        )

        await ctx.send(
            "❌ Stat invalide.\n"
            f"Statistiques disponibles : `{champs}`\n\n"
            "Exemple : `!update 219502046 t4 15`"
        )

        return

    try:

        (
            row_number,
            nom_colonne,
            nouvelle_valeur,
            ligne_finale,
        ) = await asyncio.to_thread(
            modifier_derniere_verification_google,
            player_id,
            champ_normalise,
            valeur,
        )

    except ValueError as e:

        await ctx.send(
            f"❌ {e}"
        )

        return

    except Exception as e:

        print(
            f"❌ Erreur !update : {repr(e)}"
        )

        await ctx.send(
            "❌ Impossible de modifier Google Sheets."
        )

        return

    if row_number is None:

        await ctx.send(
            f"❌ Aucune ligne trouvée pour "
            f"**{player_id}**."
        )

        return

    message = (
        "✅ **Google Sheet updated**\n\n"
        f"👤 **Player ID:** {player_id}\n"
        f"📝 **{nom_colonne}:** {nouvelle_valeur}\n"
    )

    # Afficher aussi le total si T4/T5 a été modifié.
    if champ_normalise in {
        "t4",
        "t5",
    }:

        message += (
            f"⚔️ **Total:** "
            f"{ligne_finale.get('Total', '')}\n"
        )

    message += (
        f"📄 **Sheet row:** {row_number}"
    )

    await ctx.send(
        message
    )


# =========================================================
# COMMANDE !DELETE
# =========================================================

@bot.command(
    name="delete",
)
@commande_dans_bon_salon()
@admin_ou_role_autorise()
async def delete_command(
    ctx,
    player_id: str,
):

    if not player_id.isdigit():

        await ctx.send(
            "❌ Player ID invalide.\n"
            "Exemple : `!delete 219502046`"
        )

        return

    try:

        supprimee = await asyncio.to_thread(
            supprimer_player_id_google,
            player_id,
        )

    except Exception as e:

        print(
            f"❌ Erreur !delete : {repr(e)}"
        )

        await ctx.send(
            "❌ Impossible de supprimer le joueur "
            "dans Google Sheets."
        )

        return

    if not supprimee:

        await ctx.send(
            f"❌ Aucune ligne trouvée pour "
            f"**{player_id}**."
        )

        return

    await ctx.send(
        "🗑️ **Google Sheet updated**\n\n"
        f"👤 **Player ID:** {player_id}\n"
        "✅ **1 ligne supprimée.**"
    )


# =========================================================
# COMMANDE !HELP
# =========================================================

@bot.command(
    name="rokhelp",
)
@commande_dans_bon_salon()
async def rokhelp_command(
    ctx,
):

    message = (
        "🤖 **Emi's slave 2.0 — Commands**\n\n"
        "`@Emi's slave 2.0` → statistiques globales\n\n"
        "`!latest <Player ID>` → dernière vérification\n"
        "`!history <Player ID>` → historique (10 dernières)\n\n"
        "🔢 A Player ID must contain exactly 9 digits.\n"
        "🔒 Un Player ID déjà présent dans `hopital` "
        "ne peut pas être soumis une deuxième fois. "
        "Il redevient disponible après suppression de toutes "
        "ses lignes.\n\n"
        f"`!update <Player ID> <stat> <valeur>` → "
        f"modifier la dernière ligne "
        f"(rôle **{ADMIN_ROLE_NAME}** requis)\n"
        "`!delete <Player ID>` → supprimer la vérification "
        "du joueur "
        f"(rôle **{ADMIN_ROLE_NAME}** requis)\n\n"
        "**Stats disponibles pour !update :**\n"
        "`t4`, `t5`, `total`, `food`, `wood`, `stone`, `gold`\n\n"
        "**Exemple :**\n"
        "`!update 219502046 t4 15`\n"
        "`!delete 219502046`"
    )

    await ctx.send(
        message
    )


# =========================================================
# ERREURS DES COMMANDES
# =========================================================

@bot.event
async def on_command_error(
    ctx,
    error,
):

    if isinstance(
        error,
        commands.CommandNotFound,
    ):
        return

    if isinstance(
        error,
        commands.MissingRequiredArgument,
    ):

        await ctx.send(
            "❌ Paramètre manquant.\n"
            "Utilise `!rokhelp` pour voir les commandes."
        )

        return

    if isinstance(
        error,
        commands.CheckFailure,
    ):

        await ctx.send(
            f"❌ {error}"
        )

        return

    if isinstance(
        error,
        commands.BadArgument,
    ):

        await ctx.send(
            "❌ Paramètre invalide."
        )

        return

    print(
        f"❌ Erreur de commande "
        f"{getattr(ctx.command, 'name', '?')}: "
        f"{repr(error)}"
    )

    await ctx.send(
        "❌ Une erreur inattendue est survenue."
    )


# =========================================================
# RECEPTION DES MESSAGES
# =========================================================

@bot.event
async def on_message(
    message,
):

    # -----------------------------------------------------
    # IGNORER LES MESSAGES DU BOT
    # -----------------------------------------------------

    if message.author.bot:
        return

    # -----------------------------------------------------
    # PING DU BOT -> STATISTIQUES GLOBALES
    # -----------------------------------------------------

    if (
        bot.user is not None
        and
        bot.user in message.mentions
    ):

        try:

            statistiques = await asyncio.to_thread(
                obtenir_statistiques_globales_google
            )

            joueurs = statistiques["joueurs"]
            total_t4 = statistiques["t4"]
            total_t5 = statistiques["t5"]
            puissance = statistiques["puissance"]

            await message.channel.send(
                "📊 **ROK Hospital Statistics**\n\n"
                f"👥 **Players verified:** "
                f"{formater_nombre(joueurs)}\n"
                f"🟪 **Total T4:** "
                f"{formater_nombre(total_t4)}\n"
                f"🟧 **Total T5:** "
                f"{formater_nombre(total_t5)}\n"
                f"💀 **Total power lost:** "
                f"{formater_nombre(puissance)}"
            )

        except Exception as e:

            print(
                "❌ Erreur statistiques globales : "
                f"{repr(e)}"
            )

            await message.channel.send(
                "❌ **Unable to retrieve hospital statistics "
                "from Google Sheets.**"
            )

        return

    # -----------------------------------------------------
    # SALON DE VERIFICATION
    # -----------------------------------------------------

    if message.channel.id == VERIFICATION_CHANNEL_ID:

        print()
        print(
            "========================================"
        )

        print(
            "NOUVEAU MESSAGE DE VERIFICATION"
        )

        print(
            f"Auteur : {message.author}"
        )

        print(
            f"ID     : {message.content.strip()}"
        )

        print(
            f"Images : {len(message.attachments)}"
        )

        print(
            "========================================"
        )

        result_channel = (
            await obtenir_salon_resultats()
        )

        if result_channel is None:

            print(
                "❌ Salon de résultats introuvable."
            )

            return

        # -------------------------------------------------
        # VERIFICATION DES CAPTURES
        # -------------------------------------------------

        attachments_images = []

        for attachment in message.attachments:

            extension = os.path.splitext(
                attachment.filename.lower()
            )[1]

            if extension in IMAGE_EXTENSIONS:

                attachments_images.append(
                    attachment
                )

        if not attachments_images:

            await result_channel.send(
                "❌ **Verification failed**\n"
                f"👤 **Player ID:** "
                f"`{message.content.strip() or 'Unknown'}`\n"
                "No valid screenshot was attached."
            )

            await envoyer_dm_erreur(
                message.author,
                "Verification error",
                (
                    "Your verification could not be processed "
                    "because no valid screenshot was attached. "
                    "Please send 1 or 2 hospital screenshots."
                ),
            )

            return

        if len(attachments_images) > MAX_IMAGES:

            await result_channel.send(
                "❌ **Verification failed**\n"
                f"👤 **Player ID:** "
                f"`{message.content.strip() or 'Unknown'}`\n"
                f"Maximum allowed screenshots: "
                f"**{MAX_IMAGES}**."
            )

            await envoyer_dm_erreur(
                message.author,
                "Verification error",
                (
                    f"You sent too many screenshots. "
                    f"Only {MAX_IMAGES} screenshots are allowed."
                ),
            )

            return

        # -------------------------------------------------
        # PLAYER ID
        # -------------------------------------------------

        player_id = message.content.strip()

        if not player_id.isdigit():

            await result_channel.send(
                "❌ **Verification failed**\n"
                f"👤 **Player ID:** "
                f"`{player_id or 'Unknown'}`\n"
                "The message must contain the numeric "
                "Player ID only."
            )

            await envoyer_dm_erreur(
                message.author,
                "Invalid Player ID",
                (
                    "The Player ID must contain numbers only. "
                    "Please check that you entered it correctly."
                ),
            )

            return

        if len(player_id) != 9:

            await result_channel.send(
                "❌ **Verification failed**\n"
                f"👤 **Player ID:** "
                f"`{player_id}`\n"
                "A valid Player ID must contain exactly "
                "**9 digits**."
            )

            await envoyer_dm_erreur(
                message.author,
                "Invalid Player ID",
                (
                    f"Your Player ID has **{len(player_id)} digits**, "
                    "but a valid Player ID must contain exactly "
                    "**9 digits**. Please check for a missing "
                    "or extra digit and send it again."
                ),
            )

            return

        # -------------------------------------------------
        # ANTI-DOUBLON + TRAITEMENT COMPLET
        # -------------------------------------------------
        #
        # Le verrou empêche deux personnes de soumettre le
        # même ID en même temps.
        #
        # L'ID n'est réservé que lorsque la vérification a
        # réellement été écrite dans Google Sheets.
        #
        # Un ID déjà présent dans "hopital" est refusé.
        # Après suppression de toutes ses lignes avec !delete,
        # il redevient disponible.

        async with verification_lock:

            try:

                deja_present = await asyncio.to_thread(
                    player_id_existe_google,
                    player_id,
                )

            except Exception as e:

                print(
                    "❌ ERREUR VERIFICATION DOUBLON"
                )

                print(
                    repr(e)
                )

                await result_channel.send(
                    "❌ **Verification error**\n\n"
                    f"👤 **Player ID:** "
                    f"{player_id}\n\n"
                    "The bot could not check whether "
                    "this Player ID already exists "
                    "in Google Sheets."
                )

                await envoyer_dm_erreur(
                    message.author,
                    "Verification error",
                    (
                        "The bot could not check whether your "
                        "Player ID already exists in Google Sheets. "
                        "Please try again later."
                    ),
                )

                return

            if deja_present:

                await result_channel.send(
                    "⚠️ **Player ID already verified**\n\n"
                    f"👤 **Player ID:** "
                    f"{player_id}\n\n"
                    "This Player ID already exists in "
                    "the `hopital` sheet.\n"
                    "The new verification was ignored."
                )

                await envoyer_dm_erreur(
                    message.author,
                    "Player ID already verified",
                    (
                        f"Player ID {player_id} has already been "
                        "verified and is already present in the "
                        "`hopital` sheet. Your new verification "
                        "was ignored."
                    ),
                )

                return

            # -------------------------------------------------
            # DOSSIER TEMPORAIRE
            # -------------------------------------------------

            dossier_temporaire = tempfile.mkdtemp(
                prefix="rok_hospital_"
            )

            fichiers_temporaires = []

            try:

                # -------------------------------------------------
                # TELECHARGEMENT DES CAPTURES
                # -------------------------------------------------

                for index, attachment in enumerate(
                    attachments_images,
                    start=1,
                ):

                    extension = os.path.splitext(
                        attachment.filename
                    )[1].lower()

                    if not extension:

                        extension = ".png"

                    chemin = os.path.join(
                        dossier_temporaire,
                        f"screenshot_{index}{extension}",
                    )

                    await attachment.save(
                        chemin
                    )

                    fichiers_temporaires.append(
                        chemin
                    )

                    print(
                        f"Capture {index} téléchargée -> "
                        f"{chemin}"
                    )

                # -------------------------------------------------
                # ANALYSE OCR
                # -------------------------------------------------

                print(
                    "Analyse en cours..."
                )

                resultat = await asyncio.to_thread(
                    analyser_plusieurs_images,
                    fichiers_temporaires,
                )

                print(
                    "Analyse terminée."
                )

                print(
                    resultat
                )

                t4 = resultat.get(
                    "t4"
                )

                t5 = resultat.get(
                    "t5"
                )

                total = resultat.get(
                    "total"
                )

                nourriture = resultat.get(
                    "nourriture"
                )

                bois = resultat.get(
                    "bois"
                )

                pierre = resultat.get(
                    "pierre"
                )

                or_ = resultat.get(
                    "or"
                )

                # -------------------------------------------------
                # VALIDATION
                # -------------------------------------------------

                analyse_valide = (
                    t4 is not None
                    and t5 is not None
                    and total is not None
                    and t4 + t5 == total
                )

                fichiers_discord = (
                    await envoyer_images_resultat(
                        fichiers_temporaires
                    )
                )

                if not analyse_valide:

                    await result_channel.send(
                        content=(
                            "❌ **Verification failed**\n\n"
                            f"👤 **Player ID:** "
                            f"{player_id}\n\n"
                            "The screenshots could not be "
                            "analyzed correctly.\n"
                            "The original screenshots are "
                            "attached below for manual review."
                        ),
                        files=fichiers_discord,
                    )

                    await envoyer_dm_erreur(
                        message.author,
                        "Verification error",
                        (
                            "Your hospital screenshots could not "
                            "be analyzed correctly. Please check "
                            "the screenshots and send them again."
                        ),
                    )


                    print(
                        "❌ Analyse invalide : "
                        "message source conservé."
                    )

                    return

                # -------------------------------------------------
                # GOOGLE SHEETS
                # -------------------------------------------------

                try:

                    print(
                        "Ajout de la vérification "
                        "dans Google Sheets..."
                    )

                    await asyncio.to_thread(
                        ajouter_verification_google,
                        player_id,
                        t4,
                        t5,
                        total,
                        nourriture,
                        bois,
                        pierre,
                        or_,
                    )

                except Exception as e:

                    print(
                        "❌ ERREUR GOOGLE SHEETS"
                    )

                    print(
                        repr(e)
                    )

                    await result_channel.send(
                        content=(
                            "❌ **Verification error**\n\n"
                            f"👤 **Player ID:** "
                            f"{player_id}\n\n"
                            "The verification was analyzed "
                            "correctly, but the result could "
                            "not be saved to Google Sheets.\n"
                            "The original screenshots are "
                            "attached below."
                        ),
                        files=fichiers_discord,
                    )

                    await envoyer_dm_erreur(
                        message.author,
                        "Saving error",
                        (
                            "Your hospital verification was analyzed "
                            "correctly, but the result could not be "
                            "saved to Google Sheets. Please try again "
                            "later."
                        ),
                    )


                    return

                # -------------------------------------------------
                # RESULTAT
                # -------------------------------------------------

                lignes_resultat = [

                    "✅ **Verification completed**",

                    "",

                    f"👤 **Player ID:** {player_id}",

                    "",

                    f"🟪 **T4:** {t4:,}",

                    f"🟧 **T5:** {t5:,}",

                    f"⚔️ **Total troops:** {total:,}",

                    "",

                    (
                        f"🌾 **Food:** {nourriture:,}"
                        if nourriture is not None
                        else
                        "🌾 **Food:**"
                    ),

                    (
                        f"🪵 **Wood:** {bois:,}"
                        if bois is not None
                        else
                        "🪵 **Wood:**"
                    ),

                    (
                        f"🪨 **Stone:** {pierre:,}"
                        if pierre is not None
                        else
                        "🪨 **Stone:**"
                    ),

                    (
                        f"🪙 **Gold:** {or_:,}"
                        if or_ is not None
                        else
                        "🪙 **Gold:**"
                    ),
                ]

                message_resultat = "\n".join(
                    lignes_resultat
                )

                await result_channel.send(
                    content=message_resultat,
                    files=fichiers_discord,
                )

                print(
                    "✅ Vérification envoyée "
                    "dans le salon de résultats."
                )

                # -------------------------------------------------
                # SUPPRESSION DU MESSAGE SOURCE
                # -------------------------------------------------

            except Exception as e:

                print(
                    "❌ ERREUR PENDANT LA VERIFICATION"
                )

                print(
                    repr(e)
                )

                try:

                    fichiers_discord = (
                        await envoyer_images_resultat(
                            fichiers_temporaires
                        )
                    )

                    await result_channel.send(
                        content=(
                            "❌ **Verification error**\n\n"
                            f"👤 **Player ID:** "
                            f"{player_id}\n\n"
                            "An unexpected error occurred "
                            "while analyzing the screenshots.\n"
                            "The original screenshots are "
                            "attached below for manual review."
                        ),
                        files=fichiers_discord,
                    )

                    await envoyer_dm_erreur(
                        message.author,
                        "Verification error",
                        (
                            "An unexpected error occurred while "
                            "analyzing your hospital screenshots. "
                            "Please try sending them again."
                        ),
                    )


                except Exception as send_error:

                    print(
                        "❌ Impossible d'envoyer "
                        "le rapport d'erreur : "
                        f"{send_error}"
                    )

            finally:

                try:

                    shutil.rmtree(
                        dossier_temporaire,
                        ignore_errors=True,
                    )

                    print(
                        "🧹 Fichiers temporaires supprimés."
                    )

                except Exception as e:

                    print(
                        "⚠️ Erreur nettoyage temporaire : "
                        f"{e}"
                    )

        return

    # -----------------------------------------------------
    # AUTRES SALONS -> TRAITEMENT DES COMMANDES
    # -----------------------------------------------------

    await bot.process_commands(
        message
    )


# =========================================================
# LANCEMENT
# =========================================================

if __name__ == "__main__":

    if TOKEN == "TON_TOKEN_ICI":

        raise RuntimeError(
            "Remplace TOKEN = "
            "\"TON_TOKEN_ICI\" "
            "par le token de ton bot Discord."
        )

    if not os.getenv("GOOGLE_CREDENTIALS_JSON"):

        if not os.path.exists(
            GOOGLE_CREDENTIALS_FILE
        ):

            raise FileNotFoundError(
                f"Fichier introuvable : "
                f"{GOOGLE_CREDENTIALS_FILE}\n"
                "Place le fichier JSON du compte "
                "de service dans le même dossier que bot.py, "
                "ou définis GOOGLE_CREDENTIALS_JSON."
            )

    bot.run(
        TOKEN
    )
