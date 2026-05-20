import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("starflight")

CACHE_FILE = Path(__file__).parent / "nearby_stars_cache.json"
CACHE_VERSION = 2  # increment to invalidate old cache
nearby_stars_cache: list[dict] = []


def _fetch_nearby_stars_from_vizier() -> list[dict]:
    # Plx > 2 mas  →  distance < 500 pc / ~1630 LY  (~25-35k stars from Hipparcos)
    v = Vizier(
        columns=["HIP", "Vmag", "Plx", "RAICRS", "DEICRS", "B-V"],
        row_limit=-1,
    )
    tables = v.query_constraints(catalog="I/239/hip_main", Plx=">2")
    if not tables:
        logger.warning("Vizier returned no results for Hipparcos nearby-star query")
        return []

    table = tables[0]
    stars = []
    for row in table:
        try:
            plx = float(row["Plx"])
            if plx <= 0:
                continue

            distance_pc = 1000.0 / plx
            distance_ly = distance_pc * 3.26156

            ra_deg = float(row["RAICRS"])
            dec_deg = float(row["DEICRS"])

            coord = SkyCoord(
                ra=ra_deg * u.deg,
                dec=dec_deg * u.deg,
                distance=distance_ly * u.lyr,
            )

            vmag = row["Vmag"]
            bv = row["B-V"]

            stars.append({
                "id": f"HIP{int(row['HIP'])}",
                "x": round(float(coord.cartesian.x.to(u.lyr).value), 4),
                "y": round(float(coord.cartesian.y.to(u.lyr).value), 4),
                "z": round(float(coord.cartesian.z.to(u.lyr).value), 4),
                "distance_ly": round(distance_ly, 3),
                "magnitude": round(float(vmag), 2) if not np.ma.is_masked(vmag) else None,
                "bv": round(float(bv), 3) if not np.ma.is_masked(bv) else None,
            })
        except Exception:
            continue

    stars.sort(key=lambda s: s["distance_ly"])
    return stars


def _load_cache() -> list[dict]:
    try:
        data = json.loads(CACHE_FILE.read_text())
        # Support versioned cache; legacy unversioned list is invalidated
        if isinstance(data, dict):
            if data.get("version") != CACHE_VERSION:
                logger.info("Cache version mismatch — will re-fetch")
                return []
            return data["stars"]
        logger.info("Legacy unversioned cache detected — will re-fetch")
        return []
    except Exception:
        return []


def _save_cache(stars: list[dict]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps({"version": CACHE_VERSION, "stars": stars}))
    except Exception as e:
        logger.warning(f"Could not write star cache: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nearby_stars_cache
    if CACHE_FILE.exists():
        nearby_stars_cache = _load_cache()
        if nearby_stars_cache:
            logger.info(f"Loaded {len(nearby_stars_cache)} nearby stars from cache")

    if not nearby_stars_cache:
        logger.info("Fetching nearby stars from Vizier/Hipparcos (one-time, may take ~30s)…")
        nearby_stars_cache = _fetch_nearby_stars_from_vizier()
        _save_cache(nearby_stars_cache)
        logger.info(f"Cached {len(nearby_stars_cache)} nearby stars")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Simbad.add_votable_fields("parallax", "sp_type", "V")


@app.get("/stars/nearby/status")
def nearby_status():
    return {"loaded": len(nearby_stars_cache) > 0, "count": len(nearby_stars_cache)}


@app.get("/stars/nearby")
def get_nearby_stars(max_distance_ly: float = 1500.0, limit: int = 8000,
                     cx: float = 0.0, cy: float = 0.0, cz: float = 0.0):
    if not nearby_stars_cache:
        return {"count": 0, "stars": [], "warning": "Star catalog not yet loaded"}

    r2 = max_distance_ly * max_distance_ly
    filtered = []
    for s in nearby_stars_cache:
        dx = s["x"] - cx; dy = s["y"] - cy; dz = s["z"] - cz
        if dx*dx + dy*dy + dz*dz <= r2:
            filtered.append(s)

    result = filtered[:limit]
    return {"count": len(result), "stars": result}


@app.get("/star/{name}")
def get_star(name: str):
    try:
        result = Simbad.query_object(name)

        if result is None:
            raise HTTPException(status_code=404, detail="Star not found")

        parallax = result["plx_value"][0]

        if np.ma.is_masked(parallax) or parallax <= 0:
            raise HTTPException(status_code=422, detail="No distance data for this star")

        distance_pc = 1000.0 / parallax
        distance_ly = distance_pc * 3.26156

        ra = float(result["ra"][0])
        dec = float(result["dec"][0])

        coord = SkyCoord(ra=ra, dec=dec, distance=distance_ly, unit=("deg", "deg", "lyr"))

        x = float(coord.cartesian.x.value)
        y = float(coord.cartesian.y.value)
        z = float(coord.cartesian.z.value)

        return {
            "name": name,
            "ra": ra,
            "dec": dec,
            "distance_ly": round(distance_ly, 2),
            "distance_pc": round(distance_pc, 2),
            "spectral_type": str(result["sp_type"][0]),
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
