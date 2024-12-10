import random
import string

NAMES = ["Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack",
         "2coolut", "Articlent", "BradelLuv", "Capeidge", "ChiquitaHenry", "CincoTricked", "CraziiTin", 
         "Culprol", "Ethicelle", "Fabusten", "Foloft", "Fortuneyber", "FraserDancer", "GingerGamer", "GingerPlace", 
         "Glikshe", "InspireDrummer", "Jadener", "Liventan", "Loventvae", "MarcsDot", "Nizhpoism", "PapaProphecy", 
         "Robmori", "SandHelp", "Shinannm", "Sillyti", "SlipkBugs", "Teipleth", "Witchim",
        "Agrocle", "Anthemtek", "Awmound", "Bloggest", "Bubbleahle", "CyTough", "Decksp", "Dolleyra", 
        "Faitham", "HelpfulFizz", "HippoWalker", "Latestific", "LimeWil", "NycTaru", "Plotsuse", "Promet", 
        "Proveola", "Readertapl", "RomanticChan", "SaberBigg", "Shabbyst", "SkateGrabs", "Spinanol", "Startow", 
        "Thebestiano", "Vamoomat", "Veterannexo", "Warefan", "WeirdInca", "Wzy2hot", "XenonTight", "Yahooly",
        "BabySoccer", "Bellinek", "BestHugz", "BizarreRely", "ChickSnoop", "Chikker", "ClawFalls", "CleverChase",
        "ConspiracyWar", "Cyberst", "Dancema", "Equilee", "GoldIam", "Guided", "Imassie", "Kuliker", "LikeBlab", "LiveChamp",
        "Lukeep", "MafiaChrono", "PureBrood", "ShiyaCheese", "Spuffyreda", "SraVamp", "SublimePong", "Thenorks", "Winmick",
        "WizardBoa", "YugiFashion", "ZerpAsp", "ZestyDragon"
]

def random_string(length=8):
    """Generate a random string of given length."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def random_number_string(length=4):
    """Generate a random numeric string of given length."""
    return ''.join(random.choice(string.digits) for _ in range(length))

def generate_username():
    """Generate a diverse username using various strategies."""
    strategy = random.choice(range(6))
    
    if strategy == 0:
        # Dashed names
        return random.choice(NAMES) + "-" + random_string(4)
    elif strategy == 1:
        # Human-like names with numbers
        return random.choice(NAMES) + random_number_string(2)
    elif strategy == 2:
        # Underscore-based names
        return random.choice(NAMES) + "_" + random_string(4)
    elif strategy == 3:
        # CamelCase names
        return random.choice(NAMES) + random_string(1).upper()
    elif strategy == 4:
        # Dashed names
        return random.choice(NAMES) + "-" + random.choice(NAMES)
    else:
        # Underscore-based names
        return random.choice(NAMES) + "_" + random.choice(NAMES)

def generate_diverse_usernames(n):
    """Generate n diverse random usernames."""
    usernames = set()
    while len(usernames) < n:
        usernames.add(generate_username())
    return list(usernames)