"""
generate_prehistoric_qa.py — Synthetic 30,000–10,000 BCE era text generator.

Produces factual educational passages and QA pairs about prehistoric topics.
Output schema matches sources_prehistoric/ parquet files for direct merging.

Usage:
    python pipeline/generate_prehistoric_qa.py
    python pipeline/generate_prehistoric_qa.py --n 5000 --seed 42
"""
from __future__ import annotations
import argparse, hashlib, random, time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "sources_prehistoric" / "fineweb-edu"

# ---------------------------------------------------------------------------
# Factual knowledge bases
# ---------------------------------------------------------------------------

CAVE_SITES = [
    {"name": "Lascaux",        "location": "Dordogne, France",       "date_ya": 17000, "notes": "famous for polychrome paintings of horses, aurochs, and deer"},
    {"name": "Altamira",       "location": "Cantabria, Spain",       "date_ya": 35000, "notes": "ceiling covered with bison painted in red and black ochre"},
    {"name": "Chauvet",        "location": "Ardèche, France",        "date_ya": 36000, "notes": "oldest confirmed cave art in Europe; features rhinoceros, lions, mammoths"},
    {"name": "El Castillo",    "location": "Cantabria, Spain",       "date_ya": 40800, "notes": "contains hand stencils and red discs among the oldest known cave markings"},
    {"name": "Pech Merle",     "location": "Lot, France",            "date_ya": 25000, "notes": "notable for spotted horse paintings and hand prints"},
    {"name": "Font-de-Gaume",  "location": "Dordogne, France",       "date_ya": 17000, "notes": "the last polychrome cave open to public in France"},
    {"name": "Cueva de las Manos", "location": "Santa Cruz, Argentina", "date_ya": 9000, "notes": "UNESCO site famous for hundreds of stencilled hands in red, black, and white"},
    {"name": "Blombos Cave",   "location": "Western Cape, South Africa", "date_ya": 77000, "notes": "contains earliest known abstract engravings and ochre processing kits"},
    {"name": "Pinnacle Point", "location": "Western Cape, South Africa", "date_ya": 165000, "notes": "evidence of shellfish use and ochre pigment by early Homo sapiens"},
    {"name": "Sulawesi",       "location": "Sulawesi, Indonesia",    "date_ya": 45500, "notes": "hand stencils dated to at least 45,500 years ago, among the world's oldest"},
]

TOOL_INDUSTRIES = [
    {"name": "Oldowan",       "date_ya": (2600000, 1700000), "makers": "early Homo habilis", "desc": "simple flaked pebble choppers and scrapers; the first recognizable stone tool tradition"},
    {"name": "Acheulean",     "date_ya": (1700000, 300000),  "makers": "Homo erectus and later hominins", "desc": "characterized by large teardrop-shaped handaxes and cleavers showing deliberate symmetry"},
    {"name": "Mousterian",    "date_ya": (300000, 30000),    "makers": "Neanderthals and early modern humans", "desc": "Levallois prepared-core technique producing sharp flakes and points for hafted tools"},
    {"name": "Aurignacian",   "date_ya": (43000, 28000),     "makers": "anatomically modern Homo sapiens", "desc": "blade-based technology with bone points, ivory beads, and associated cave art"},
    {"name": "Gravettian",    "date_ya": (33000, 21000),     "makers": "Homo sapiens in Europe", "desc": "backed bladelets, Venus figurines, communal mammoth hunting, and pit-house settlements"},
    {"name": "Solutrean",     "date_ya": (22000, 17000),     "makers": "Homo sapiens in France and Iberia", "desc": "finest flint-knapping of the Paleolithic, producing thin laurel-leaf and shouldered points"},
    {"name": "Magdalenian",   "date_ya": (17000, 11000),     "makers": "Homo sapiens in Western Europe", "desc": "harpoons, needles, spear-throwers (atlatls), and prolific cave art including Lascaux"},
    {"name": "Natufian",      "date_ya": (14500, 11500),     "makers": "Homo sapiens in the Levant", "desc": "semi-sedentary villages, intensive wild cereal harvesting, and early dog domestication"},
]

HOMININS = [
    {"name": "Homo habilis",          "mya": (2.8, 1.5),   "region": "East and South Africa",            "brain_cc": 550,  "notes": "first Homo; associated with Oldowan tools"},
    {"name": "Homo erectus",          "mya": (1.9, 0.1),   "region": "Africa, Asia, and possibly Europe","brain_cc": 900,  "notes": "first hominin to leave Africa; controlled fire; Acheulean tools"},
    {"name": "Homo heidelbergensis",  "mya": (0.7, 0.2),   "region": "Africa and Europe",                "brain_cc": 1200, "notes": "hunted large game; possible common ancestor of Neanderthals and modern humans"},
    {"name": "Homo neanderthalensis", "mya": (0.4, 0.04),  "region": "Europe and western Asia",          "brain_cc": 1410, "notes": "buried their dead, used pigment, interbred with Homo sapiens"},
    {"name": "Denisovans",            "mya": (0.5, 0.05),  "region": "Siberia, Tibet, Southeast Asia",   "brain_cc": None, "notes": "known mainly from DNA; interbred with both Neanderthals and Homo sapiens"},
    {"name": "Homo sapiens",          "mya": (0.3, 0.0),   "region": "global",                           "brain_cc": 1350, "notes": "anatomically modern humans; behavioural modernity by ~50,000 years ago"},
    {"name": "Homo floresiensis",     "mya": (0.1, 0.05),  "region": "Flores, Indonesia",                "brain_cc": 380,  "notes": "island dwarfism; made stone tools despite small brain size"},
]

MIGRATION_EVENTS = [
    {"event": "First Out-of-Africa dispersal",           "date_ya": 1_900_000, "route": "Homo erectus via the Levant into Eurasia",               "notes": "reached China and Java within a few hundred thousand years"},
    {"event": "Modern human expansion from Africa",      "date_ya": 70_000,   "route": "across the Red Sea and Horn of Africa into Arabia",       "notes": "possibly triggered by the Toba supervolcano eruption ~74,000 ya"},
    {"event": "Peopling of Australia",                   "date_ya": 65_000,   "route": "island-hopping through Southeast Asia",                   "notes": "required sea crossings; earliest evidence at Madjedbebe rock shelter"},
    {"event": "Arrival in Europe",                       "date_ya": 45_000,   "route": "through the Levant and Anatolia",                         "notes": "overlapped with Neanderthals for 5,000–10,000 years"},
    {"event": "Beringian crossing",                      "date_ya": 20_000,   "route": "across Beringia land bridge from Siberia to Alaska",      "notes": "Beringia exposed when sea levels fell ~120 m during glacial maximum"},
    {"event": "Peopling of the Americas",                "date_ya": 15_000,   "route": "south along Pacific coast and through ice-free corridor",  "notes": "Monte Verde site in Chile dates to ~14,800 ya"},
    {"event": "Colonisation of Pacific islands (early)", "date_ya": 35_000,   "route": "into Melanesia and near Oceania",                         "notes": "reached the Solomon Islands using watercraft"},
]

ARCHAEOLOGICAL_SITES = [
    {"name": "Göbekli Tepe",    "location": "Şanlıurfa, Turkey",          "date_ya": 11500, "significance": "world's oldest known monumental structures; T-shaped limestone pillars carved with animals; predates agriculture"},
    {"name": "Jericho",         "location": "West Bank, Palestine",        "date_ya": 11000, "significance": "one of the earliest continuously inhabited towns; Natufian and Pre-Pottery Neolithic layers"},
    {"name": "Çatalhöyük",      "location": "Konya Plain, Turkey",         "date_ya": 9000,  "significance": "densely packed Neolithic town of ~8,000 people; rich in wall paintings and figurines"},
    {"name": "Dolní Věstonice", "location": "South Moravia, Czechia",      "date_ya": 29000, "significance": "earliest known fired ceramic figurines including the Venus of Dolní Věstonice"},
    {"name": "Sungir",          "location": "Vladimir Oblast, Russia",      "date_ya": 34000, "significance": "Gravettian burials with thousands of mammoth-ivory beads; evidence of complex mortuary ritual"},
    {"name": "Pincevent",       "location": "Seine-et-Marne, France",      "date_ya": 12000, "significance": "Magdalenian hunter-gatherer camp with tent floors, hearths, and reindeer butchery evidence"},
    {"name": "Star Carr",       "location": "North Yorkshire, England",     "date_ya": 11000, "significance": "Mesolithic site; antler headdresses suggest shamanic ritual; earliest British carpentry"},
    {"name": "Ohalo II",        "location": "Sea of Galilee, Israel",       "date_ya": 23000, "significance": "brush-hut settlement; earliest evidence of bread-making from wild grass seeds"},
    {"name": "Blombos Cave",    "location": "Western Cape, South Africa",   "date_ya": 77000, "significance": "ochre engravings, shell beads, and compound adhesive — early behavioural modernity"},
    {"name": "Trinil",          "location": "East Java, Indonesia",         "date_ya": 700000,"significance": "discovery site of Homo erectus fossils by Eugène Dubois in 1891"},
]

VENUS_FIGURINES = [
    {"name": "Venus of Willendorf",    "location": "Willendorf, Austria",       "date_ya": 30000, "material": "oolitic limestone",      "height_cm": 11.1},
    {"name": "Venus of Hohle Fels",    "location": "Swabian Jura, Germany",     "date_ya": 40000, "material": "woolly mammoth ivory",   "height_cm": 6.0},
    {"name": "Venus of Laussel",       "location": "Dordogne, France",          "date_ya": 25000, "material": "limestone bas-relief",   "height_cm": 44.0},
    {"name": "Venus of Dolní Věstonice","location": "South Moravia, Czechia",   "date_ya": 29000, "material": "fired clay",             "height_cm": 11.5},
    {"name": "Venus of Lespugue",      "location": "Haute-Garonne, France",     "date_ya": 26000, "material": "ivory",                  "height_cm": 14.7},
    {"name": "Venus of Brassempouy",   "location": "Landes, France",            "date_ya": 25000, "material": "mammoth ivory",          "height_cm": 3.65},
]

CLIMATE_EVENTS = [
    {"event": "Last Glacial Maximum (LGM)",         "date_ya": 26500, "description": "ice sheets covered much of North America, Europe, and Asia; sea levels were ~120 m lower than today; Sahara was hyper-arid"},
    {"event": "Toba supervolcano eruption",         "date_ya": 74000, "description": "largest volcanic eruption in 2 million years; possibly caused a 6–10 year volcanic winter reducing Homo sapiens to a small founder population"},
    {"event": "Heinrich Event 1",                   "date_ya": 17000, "description": "massive iceberg discharge into the North Atlantic disrupting ocean circulation and causing rapid cooling in Europe"},
    {"event": "Bølling–Allerød warm period",        "date_ya": 14700, "description": "abrupt warming of ~10°C in Greenland; expansion of forests in Europe; associated with Magdalenian cultural florescence"},
    {"event": "Younger Dryas cold period",          "date_ya": 12900, "description": "sudden return to near-glacial conditions lasting ~1,200 years; triggered widespread megafauna extinctions and cultural shifts"},
    {"event": "Holocene onset (end of Ice Age)",    "date_ya": 11700, "description": "rapid warming marks end of Pleistocene; sea levels rise ~120 m over several millennia; conditions allow agriculture"},
]

MEGAFAUNA = [
    {"name": "Woolly mammoth",        "species": "Mammuthus primigenius",   "extinction_ya": 10000, "range": "North America, Europe, and northern Asia; dwarf population survived on Wrangel Island until 4,000 ya"},
    {"name": "Woolly rhinoceros",     "species": "Coelodonta antiquitatis", "extinction_ya": 10000, "range": "Eurasian steppe; depicted in Chauvet Cave paintings"},
    {"name": "Cave lion",             "species": "Panthera spelaea",        "extinction_ya": 13000, "range": "Europe and northern Asia; depicted in Lascaux and Chauvet caves"},
    {"name": "Cave bear",             "species": "Ursus spelaeus",          "extinction_ya": 24000, "range": "Europe and Asia; ceremonially significant to Neanderthals"},
    {"name": "Giant ground sloth",    "species": "Megatherium americanum",  "extinction_ya": 10000, "range": "South America; car-sized herbivore; may have been hunted by early Americans"},
    {"name": "Saber-toothed cat",     "species": "Smilodon fatalis",        "extinction_ya": 10000, "range": "North and South America; canine teeth up to 28 cm long"},
    {"name": "Irish elk",             "species": "Megaloceros giganteus",   "extinction_ya": 7700,  "range": "Eurasia; antler span up to 3.7 m"},
    {"name": "Glyptodon",             "species": "Glyptodon sp.",           "extinction_ya": 10000, "range": "South America; armadillo-like herbivore the size of a Volkswagen Beetle"},
]

# ---------------------------------------------------------------------------
# Passage templates
# ---------------------------------------------------------------------------

def passage_cave_site(site: dict, rng: random.Random) -> str:
    alt_verbs = ["is renowned for", "is celebrated for", "is noted for", "stands out for"]
    verb = rng.choice(alt_verbs)
    age_phrase = f"approximately {site['date_ya']:,} years ago"
    return (
        f"{site['name']} is a prehistoric cave site located in {site['location']}. "
        f"Dated to {age_phrase}, it {verb} {site['notes']}. "
        f"The site provides invaluable evidence of the symbolic and artistic capabilities "
        f"of Upper Paleolithic Homo sapiens, who used natural pigments such as ochre, "
        f"charcoal, and haematite to create images on the cave walls. "
        f"Cave art sites like {site['name']} demonstrate that behavioural modernity — "
        f"including abstract thought, planning, and cultural transmission — was fully "
        f"established in our species during the late Pleistocene epoch."
    )


def passage_tool_industry(ind: dict, rng: random.Random) -> str:
    start, end = ind["date_ya"]
    duration = start - end
    return (
        f"The {ind['name']} industry is a stone tool tradition associated with "
        f"{ind['makers']}, spanning roughly {start:,} to {end:,} years ago "
        f"(a period of about {duration:,} years). "
        f"It is characterised by {ind['desc']}. "
        f"The technological sophistication of the {ind['name']} tradition reflects "
        f"the cognitive and adaptive capacities of the hominins who produced it, "
        f"offering archaeologists a window into prehistoric problem-solving, "
        f"resource procurement strategies, and social learning across generations."
    )


def passage_hominin(h: dict, rng: random.Random) -> str:
    start_mya, end_mya = h["mya"]
    brain_str = f"a cranial capacity averaging around {h['brain_cc']} cc" if h["brain_cc"] else "a cranial capacity not yet precisely established from fossil evidence"
    return (
        f"{h['name']} lived from approximately {start_mya} to {end_mya} million years ago "
        f"across {h['region']}. With {brain_str}, "
        f"this hominin species is significant because {h['notes']}. "
        f"The study of {h['name']} contributes to our understanding of human evolution "
        f"during the Pleistocene, illuminating the mosaic nature of anatomical and "
        f"behavioural change across the hominin lineage."
    )


def passage_migration(ev: dict, rng: random.Random) -> str:
    return (
        f"The {ev['event']} occurred approximately {ev['date_ya']:,} years ago. "
        f"Populations moved {ev['route']}. "
        f"{ev['notes'].capitalize()}. "
        f"This dispersal event is reconstructed from a convergence of archaeological, "
        f"genetic, and palaeoclimatic evidence, demonstrating the remarkable adaptability "
        f"of early Homo sapiens to diverse and often challenging environments during "
        f"the late Pleistocene."
    )


def passage_site(site: dict, rng: random.Random) -> str:
    return (
        f"{site['name']} is an archaeological site in {site['location']}, "
        f"dated to approximately {site['date_ya']:,} years ago. "
        f"It is significant because {site['significance']}. "
        f"Excavations at {site['name']} have been pivotal in reshaping our understanding "
        f"of prehistoric human behaviour, social organisation, and the pace of cultural "
        f"innovation during the transition from mobile hunter-gatherer lifestyles toward "
        f"the more complex social arrangements that preceded agriculture."
    )


def passage_venus(v: dict, rng: random.Random) -> str:
    return (
        f"The {v['name']} is a Gravettian-period figurine discovered at "
        f"{v['location']}, dated to approximately {v['date_ya']:,} years ago. "
        f"Carved from {v['material']} and measuring {v['height_cm']} cm in height, "
        f"it belongs to a widespread class of prehistoric female figurines found "
        f"across Europe. Venus figurines are among the earliest known examples of "
        f"portable art, and their cross-continental distribution suggests shared "
        f"symbolic or ritual traditions among Upper Paleolithic hunter-gatherer groups "
        f"during the Last Glacial Maximum."
    )


def passage_climate(ev: dict, rng: random.Random) -> str:
    return (
        f"The {ev['event']} occurred approximately {ev['date_ya']:,} years ago. "
        f"{ev['description'].capitalize()}. "
        f"Palaeoclimatic events of this magnitude profoundly shaped human prehistory "
        f"by altering resource availability, driving population movements, and selecting "
        f"for new technological and social strategies. Understanding these climate episodes "
        f"is essential for contextualising the archaeological record of the Pleistocene "
        f"and the transitions that led to the emergence of complex societies."
    )


def passage_megafauna(m: dict, rng: random.Random) -> str:
    return (
        f"The {m['name']} ({m['species']}) went extinct approximately "
        f"{m['extinction_ya']:,} years ago. Its range encompassed {m['range']}. "
        f"The late Pleistocene megafauna extinction event eliminated the majority of "
        f"large-bodied mammals worldwide. Debate continues over the relative contributions "
        f"of climate change following the Last Glacial Maximum and human overkill hunting "
        f"pressure. The disappearance of megafauna like the {m['name']} had cascading "
        f"ecological consequences and forced prehistoric human populations to adapt their "
        f"subsistence strategies, contributing to the eventual turn toward agriculture."
    )


def qa_cave_site(site: dict, rng: random.Random) -> str:
    q = rng.choice([
        f"Where is the prehistoric cave site of {site['name']} located?",
        f"How old is the cave art found at {site['name']}?",
        f"What makes {site['name']} significant in the study of prehistoric art?",
    ])
    a = f"{site['name']} is located in {site['location']}. It dates to approximately {site['date_ya']:,} years ago and {site['notes']}."
    return f"Question: {q}\nAnswer: {a}"


def qa_tool(ind: dict, rng: random.Random) -> str:
    q = rng.choice([
        f"What is the {ind['name']} stone tool tradition?",
        f"Who made {ind['name']} tools and when?",
        f"What characterises the {ind['name']} industry?",
    ])
    start, end = ind["date_ya"]
    a = f"The {ind['name']} industry was created by {ind['makers']}, dating from roughly {start:,} to {end:,} years ago. It is {ind['desc']}."
    return f"Question: {q}\nAnswer: {a}"


def qa_hominin(h: dict, rng: random.Random) -> str:
    q = rng.choice([
        f"What is known about {h['name']}?",
        f"Where did {h['name']} live and what is notable about this species?",
        f"When did {h['name']} exist?",
    ])
    start, end = h["mya"]
    a = f"{h['name']} lived from about {start} to {end} million years ago in {h['region']}. {h['notes'].capitalize()}."
    return f"Question: {q}\nAnswer: {a}"


def qa_site(site: dict, rng: random.Random) -> str:
    q = rng.choice([
        f"What is the significance of {site['name']}?",
        f"Where is {site['name']} located and how old is it?",
        f"Why is {site['name']} important to prehistoric archaeology?",
    ])
    a = f"{site['name']} is in {site['location']}, dated to ~{site['date_ya']:,} years ago. {site['significance']}."
    return f"Question: {q}\nAnswer: {a}"


def qa_climate(ev: dict, rng: random.Random) -> str:
    q = f"What was the {ev['event']} and when did it occur?"
    a = f"The {ev['event']} occurred approximately {ev['date_ya']:,} years ago. {ev['description']}."
    return f"Question: {q}\nAnswer: {a}"


def qa_megafauna(m: dict, rng: random.Random) -> str:
    q = rng.choice([
        f"When did the {m['name']} go extinct?",
        f"What was the range of the {m['name']}?",
    ])
    a = f"The {m['name']} ({m['species']}) went extinct about {m['extinction_ya']:,} years ago. {m['range'].capitalize()}."
    return f"Question: {q}\nAnswer: {a}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATORS = [
    (CAVE_SITES,          passage_cave_site,  qa_cave_site),
    (TOOL_INDUSTRIES,     passage_tool_industry, qa_tool),
    (HOMININS,            passage_hominin,    qa_hominin),
    (MIGRATION_EVENTS,    passage_migration,  None),
    (ARCHAEOLOGICAL_SITES,passage_site,       qa_site),
    (VENUS_FIGURINES,     passage_venus,      None),
    (CLIMATE_EVENTS,      passage_climate,    qa_climate),
    (MEGAFAUNA,           passage_megafauna,  qa_megafauna),
]


def _dedup_hash(text: str) -> str:
    return hashlib.sha256(text[:500].encode("utf-8", errors="replace")).hexdigest()


def _make_record(text: str, idx: int) -> dict:
    return {
        "text": text,
        "id":   f"synthetic_prehistoric_{idx:06d}",
        "dump": "synthetic-prehistoric-v1",
        "url":  "",
        "file_path": "synthetic/prehistoric_qa",
        "language": "en",
        "language_score": 1.0,
        "token_count": len(text.split()),
        "score": None,
        "int_score": None,
        "fk_score": None,
        "est_token_count": int(len(text.split()) * 1.3),
        "grade_level": None,
        "age_range_min": None,
        "age_range_max": None,
        "edu_score": None,
        "edu_score_estimated": True,
        "dedup_hash": _dedup_hash(text),
        "domain_slug": "anthropology",
        "domain_name": "Anthropology",
        "domain_category": "Social Sciences",
        "domain_confidence": 1.0,
        "age_epoch_of_history": "Prehistoric Era",
        "earliest_historical_year": -30000,
        "corrected_era": "prehistoric",
        "corrected_date": -30000,
    }


def generate(n: int, seed: int, out_dir: Path, max_per_shard: int) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    idx = 0

    # Build passage + QA pairs until we hit n
    while idx < n:
        items, pass_fn, qa_fn = rng.choice(GENERATORS)
        item = rng.choice(items)

        # Passage
        try:
            text = pass_fn(item, rng)
            records.append(_make_record(text, idx)); idx += 1
        except Exception:
            pass

        # QA (if generator exists)
        if qa_fn and idx < n:
            try:
                text = qa_fn(item, rng)
                records.append(_make_record(text, idx)); idx += 1
            except Exception:
                pass

    # Write shards
    shard_idx = 0
    for start in range(0, len(records), max_per_shard):
        chunk = records[start:start + max_per_shard]
        cols  = {k: [r.get(k) for r in chunk] for k in chunk[0].keys()}
        path  = out_dir / f"synthetic_prehistoric_{shard_idx:04d}.parquet"
        pq.write_table(pa.table(cols), path, compression="snappy")
        print(f"  → wrote {path.name}  ({len(chunk):,} rows)")
        shard_idx += 1

    print(f"\n  ✅ {len(records):,} synthetic records written to {out_dir}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic prehistoric QA/passages")
    ap.add_argument("--n",             type=int, default=10_000)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--max-per-shard", type=int, default=5_000)
    ap.add_argument("--out-dir",       default=str(OUT_DIR))
    args = ap.parse_args()

    print(f"\n[generate_prehistoric_qa]  n={args.n:,}  seed={args.seed}")
    t0 = time.time()
    generate(args.n, args.seed, Path(args.out_dir), args.max_per_shard)
    print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
