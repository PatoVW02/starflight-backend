import json
import logging
import math
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("starflight")

CACHE_FILE = Path(__file__).parent / "nearby_stars_cache.json"
CACHE_VERSION = 2
nearby_stars_cache: list[dict] = []

SIMBAD_TAP = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"


def _ra_dec_to_xyz(ra_deg: float, dec_deg: float, distance_ly: float) -> tuple[float, float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = distance_ly * math.cos(dec) * math.cos(ra)
    y = distance_ly * math.cos(dec) * math.sin(ra)
    z = distance_ly * math.sin(dec)
    return x, y, z


def _load_cache() -> list[dict]:
    try:
        data = json.loads(CACHE_FILE.read_text())
        if isinstance(data, dict):
            if data.get("version") != CACHE_VERSION:
                logger.info("Cache version mismatch — skipping")
                return []
            return data["stars"]
        return []
    except Exception:
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nearby_stars_cache
    nearby_stars_cache = _load_cache()
    if nearby_stars_cache:
        logger.info(f"Loaded {len(nearby_stars_cache)} nearby stars from cache")
    else:
        logger.warning("No star cache found — /stars/nearby will return empty results")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stars/nearby/status")
def nearby_status():
    return {"loaded": len(nearby_stars_cache) > 0, "count": len(nearby_stars_cache)}


@app.get("/stars/nearby")
def get_nearby_stars(
    max_distance_ly: float = 1500.0,
    limit: int = 8000,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
):
    if not nearby_stars_cache:
        return {"count": 0, "stars": [], "warning": "Star catalog not yet loaded"}

    r2 = max_distance_ly * max_distance_ly
    filtered = []
    for s in nearby_stars_cache:
        dx = s["x"] - cx
        dy = s["y"] - cy
        dz = s["z"] - cz
        if dx * dx + dy * dy + dz * dz <= r2:
            filtered.append(s)

    result = filtered[:limit]
    return {"count": len(result), "stars": result}


@app.get("/star/{name}")
async def get_star(name: str):
    # Escape single quotes for ADQL
    safe_name = name.replace("'", "''")
    adql = (
        "SELECT TOP 1 ra, dec, plx_value, sp_type "
        "FROM basic JOIN ident ON ident.oidref = basic.oid "
        f"WHERE ident.id = '{safe_name}'"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                SIMBAD_TAP,
                params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable/td", "QUERY": adql},
            )
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"SIMBAD unreachable: {exc}")

    # Parse VOTable with regex — avoids pulling in an XML library for namespace handling
    text = resp.text
    fields = re.findall(r'<FIELD[^>]+name="([^"]+)"', text)
    tr = re.search(r"<TR>(.*?)</TR>", text, re.DOTALL)
    if not tr or not fields:
        raise HTTPException(status_code=404, detail="Star not found")

    values = re.findall(r"<TD>([^<]*)</TD>", tr.group(1))
    row = dict(zip(fields, values))

    try:
        plx = float(row.get("plx_value") or 0)
    except ValueError:
        plx = 0.0

    if plx <= 0:
        raise HTTPException(status_code=422, detail="No parallax / distance data for this star")

    try:
        ra = float(row["ra"])
        dec = float(row["dec"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Incomplete coordinate data: {exc}")

    distance_pc = 1000.0 / plx
    distance_ly = distance_pc * 3.26156
    x, y, z = _ra_dec_to_xyz(ra, dec, distance_ly)

    return {
        "name": name,
        "ra": ra,
        "dec": dec,
        "distance_ly": round(distance_ly, 2),
        "distance_pc": round(distance_pc, 2),
        "spectral_type": row.get("sp_type", ""),
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
    }
