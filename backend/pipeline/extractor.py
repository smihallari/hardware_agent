from dataclasses import dataclass
import hashlib
import re

# Reject boxes, coolers, waterblocks, full PCs/laptops, and blatantly dead hardware
JUNK_PATTERNS = re.compile(
    r"\b("
    r"box only|empty box|only box|ovp leer|leerkarton|nur karton|nur ovp|bo[îi]te vide|solo scatola|"
    r"cooler only|heatsink only|fan only|backplate|waterblock|ekwb|alphacool|bykski|riser|bracket|cable|adapter|"
    r"for parts|not working|no display|artifact|code 43|broken|defective|"
    r"defekt|bastler|teildefekt|ohne funktion|f[üu]r bastler|"
    r"hs|hors service|pour pi[èe]ces|ne fonctionne pas|panne|"
    r"guasto|non funzionante|per pezzi|ricambi|"
    r"laptop|notebook|gaming pc|komplett\s?pc|pc gamer|desktop pc|workstation|barebone|"
    r"ζητειται|ζητείται|μόνο κουτί|για ανταλλακτικά"
    r")\b",
    re.IGNORECASE,
)
JUNK_TITLE_PATTERNS = re.compile(
    r"\b("
    r"dissipatore|heatsink|shroud|cooler|k[üu]hler|l[üu]fter|ventilateur|radiateur|"
    r"pcb|senza\s+viti|parti\s+per|ohne\s+(?:chip|gpu|kern|vram)"
    r")\b",
    re.IGNORECASE,
)

# Ordered from most specific to least specific so Ti/Super never match base SKUs
SKU_PATTERNS: list[tuple[re.Pattern[str], str, int]] = [
    (re.compile(r"\brtx\s*4070\s*ti\s*super\b", re.I), "RTX 4070 TI SUPER 16GB", 16),
    (re.compile(r"\brtx\s*4070\s*ti\b", re.I), "RTX 4070 TI 12GB", 12),
    (re.compile(r"\brtx\s*4070\s*super\b", re.I), "RTX 4070 SUPER 12GB", 12),
    (re.compile(r"\brtx\s*4070\b", re.I), "RTX 4070 12GB", 12),
    (re.compile(r"\brtx\s*3080\s*ti\b", re.I), "RTX 3080 TI 12GB", 12),
    (re.compile(r"\brtx\s*3080\b.*?\b12\s?g(?:b|o)\b|\b12\s?g(?:b|o)\b.*?\brtx\s*3080\b", re.I), "RTX 3080 12GB", 12),
    (re.compile(r"\brtx\s*3080\b", re.I), "RTX 3080 10GB", 10),
    (re.compile(r"\brtx\s*3070\s*ti\b", re.I), "RTX 3070 TI 8GB", 8),
    (re.compile(r"\brtx\s*3070\b", re.I), "RTX 3070 8GB", 8),
    (re.compile(r"\brtx\s*3060\s*ti\b", re.I), "RTX 3060 TI 8GB", 8),
    (re.compile(r"\brtx\s*3060\b.*?\b8\s?g(?:b|o)\b|\b8\s?g(?:b|o)\b.*?\brtx\s*3060\b", re.I), "RTX 3060 8GB", 8),
    (re.compile(r"\brtx\s*3060\b", re.I), "RTX 3060 12GB", 12),
    (re.compile(r"\brx\s*6800\s*xt\b", re.I), "RX 6800 XT 16GB", 16),
    (re.compile(r"\brx\s*6800\b", re.I), "RX 6800 16GB", 16),
    (re.compile(r"\brx\s*6750\s*xt\b", re.I), "RX 6750 XT 12GB", 12),
    (re.compile(r"\brx\s*6700\s*xt\b", re.I), "RX 6700 XT 12GB", 12),
]

FE_REF_RE = re.compile(r"\b(founders\s*edition|\bfe\b|reference|referenz)\b", re.I)
FLAGSHIP_RE = re.compile(
    r"\b(strix|suprim|aorus\s*(master|xtreme)|nitro\+|red\s*devil|ftw3|amp\s*extreme|hof|taichi|toxic|ichill|vapor-x)\b",
    re.I,
)
MID_RE = re.compile(
    r"\b(tuf|gaming\s*x|gaming\s*oc|gaming\s*z|pulse|hellhound|xc3|merc\s*319|trinity|proart|aero|vision|fighter)\b",
    re.I,
)
BOX_RE = re.compile(r"\b(ovp|original\s*box|with\s*box|w/\s*box|boxed|bo[îi]te\s*d'origine|scatola\s*originale)\b", re.I)
WARRANTY_RE = re.compile(r"\b(warranty|garantie|rechnung|invoice|facture|garanzia|fattura)\b", re.I)
MINED_RE = re.compile(r"\b(mining|mined|rig|repaste|repasted|w[äa]rmeleitpaste|repaired|refurbished|general[üu]berholt)\b", re.I)


@dataclass
class ExtractedHardware:
    content_hash: str
    category: str
    canonical_sku: str
    brand_tier: str
    vram_or_ram_gb: int | None
    storage_gb: int | None
    condition_score: int
    has_warranty: bool
    has_original_box: bool
    is_mined_or_repaired: bool
    is_valid_hardware: bool


def compute_content_hash(title: str, description: str) -> str:
    normalized = re.sub(r"\s+", " ", f"{title} {description}".lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _map_condition_score(condition_id: str | None, text: str) -> int:
    cid = str(condition_id or "")
    if cid in {"1000", "1500"}:
        return 4
    if cid in {"2000", "2010", "2020", "2500", "2990"}:
        return 3
    if cid == "3010":
        return 1
    if cid == "7000":
        return 0
    if re.search(r"\b(like\s*new|mint|neuwertig|wie\s*neu|comme\s*neuf|come\s*nuovo)\b", text, re.I):
        return 3
    return 2  # Default standard Used (3000)


def extract_deterministic(
    content_hash: str,
    title: str,
    description: str,
    asking_price: float,
    condition_id: str | None = None,
) -> ExtractedHardware:
    text = f"{title} {description or ''}"

    if asking_price < 60.0 or JUNK_PATTERNS.search(text) or JUNK_TITLE_PATTERNS.search(title):
        return ExtractedHardware(
            content_hash=content_hash,
            category="ACCESSORY_JUNK",
            canonical_sku="UNKNOWN",
            brand_tier="LOW",
            vram_or_ram_gb=None,
            storage_gb=None,
            condition_score=0,
            has_warranty=False,
            has_original_box=False,
            is_mined_or_repaired=False,
            is_valid_hardware=False,
        )

    matched_sku = None
    matched_vram = None
    for pattern, sku_name, vram in SKU_PATTERNS:
        if pattern.search(title): 
            matched_sku = sku_name
            matched_vram = vram
            break

    if not matched_sku:
        return ExtractedHardware(
            content_hash=content_hash,
            category="ACCESSORY_JUNK",
            canonical_sku="UNSUPPORTED_SKU",
            brand_tier="LOW",
            vram_or_ram_gb=None,
            storage_gb=None,
            condition_score=2,
            has_warranty=False,
            has_original_box=False,
            is_mined_or_repaired=False,
            is_valid_hardware=False,
        )

    if FE_REF_RE.search(title):
        tier = "FE_REF"
    elif FLAGSHIP_RE.search(title):
        tier = "FLAGSHIP"
    elif MID_RE.search(title):
        tier = "MID"
    else:
        tier = "LOW"

    return ExtractedHardware(
        content_hash=content_hash,
        category="GPU",
        canonical_sku=matched_sku,
        brand_tier=tier,
        vram_or_ram_gb=matched_vram,
        storage_gb=None,
        condition_score=_map_condition_score(condition_id, text),
        has_warranty=bool(WARRANTY_RE.search(text)),
        has_original_box=bool(BOX_RE.search(text)),
        is_mined_or_repaired=bool(MINED_RE.search(text)),
        is_valid_hardware=True,
    )