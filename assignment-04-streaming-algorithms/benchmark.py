"""Compare Bloom filters and Count-Min sketches on a MovieLens rating stream."""
from array import array
from collections import Counter
from hashlib import blake2b, file_digest
from pathlib import Path
import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import struct
import sys
import tempfile
from time import perf_counter
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
ARCHIVE = ROOT / "data" / "ml-1m.zip"
RESULTS = ROOT / "results"
SEED = 42
NEGATIVE_QUERIES = 10_000
BLOOM_PARAMS = [(10_000, 4), (40_000, 1), (40_000, 4), (40_000, 8), (40_000, 16), (80_000, 4)]
CMS_PARAMS = [(128, 4), (512, 1), (512, 2), (512, 4), (512, 8), (2048, 4), (4096, 4)]


class HashWords:
    """Generate reproducible 32-bit hash words for nonnegative integer IDs."""

    __slots__ = ("key", "unpack")

    def __init__(self, count, domain):
        if not 1 <= count <= 16:
            raise ValueError("Hash word count must be between 1 and 16.")
        self.key = domain + SEED.to_bytes(4, "little")
        self.unpack = struct.Struct(f"<{count}I")

    def words(self, item):
        # Fix digest length so adding hashes preserves all previous positions.
        digest = blake2b(struct.pack("<Q", item), key=self.key).digest()
        return self.unpack.unpack_from(digest)


class BloomFilter:
    """Append-only membership sketch with a packed bit array."""

    __slots__ = ("m", "bits", "hashes")

    def __init__(self, m, k):
        if m <= 0:
            raise ValueError("The bit budget must be positive.")
        self.m = m
        self.bits = bytearray((m + 7) // 8)
        self.hashes = HashWords(k, b"bloom")

    def add(self, item):
        for word in self.hashes.words(item):
            bit = word % self.m
            self.bits[bit >> 3] |= 1 << (bit & 7)

    def contains(self, item):
        for word in self.hashes.words(item):
            bit = word % self.m
            if not self.bits[bit >> 3] & (1 << (bit & 7)):
                return False
        return True


class CountMinSketch:
    """Point-frequency estimates for unit increments with unsigned counters."""

    __slots__ = ("width", "offsets", "counts", "hashes")

    def __init__(self, width, depth):
        if width <= 0:
            raise ValueError("The row width must be positive.")
        self.hashes = HashWords(depth, b"cms")
        self.width = width
        self.offsets = range(0, width * depth, width)
        self.counts = array("I", [0]) * (width * depth)

    def add(self, item):
        for offset, word in zip(self.offsets, self.hashes.words(item)):
            self.counts[offset + word % self.width] += 1

    def estimate(self, item):
        words = self.hashes.words(item)
        return min(self.counts[offset + word % self.width]
                   for offset, word in zip(self.offsets, words))


def prepare(archive):
    if archive.is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as target:
        temp = Path(target.name)
        try:
            with urlopen(URL, timeout=60) as response:
                shutil.copyfileobj(response, target)
            target.close()
            verify_archive(temp)
            temp.replace(archive)
        except URLError as error:
            raise OSError(f"Download failed. Provide the official ZIP with --archive. {error}") from error
        finally:
            target.close()
            temp.unlink(missing_ok=True)


def verify_archive(archive):
    with archive.open("rb") as source:
        sha256 = file_digest(source, "sha256").hexdigest()
    if sha256 != EXPECTED_SHA256:
        raise ValueError("ZIP SHA-256 does not match the official MovieLens 1M archive.")
    return sha256


def movie_ids(archive):
    with ZipFile(archive) as data, data.open("ml-1m/ratings.dat") as rows:
        for row in rows:
            # user ID :: movie ID :: rating :: epoch timestamp
            yield int(row.split(b"::", 2)[1])


def footprint(root):
    """Retained Python object sizes, with repeated references counted once."""
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            for key, frequency in value.items():
                total += size(key) + size(frequency)
        elif isinstance(value, set):
            total += sum(size(item) for item in value)
        elif isinstance(value, (BloomFilter, CountMinSketch, HashWords)):
            total += sum(size(getattr(value, attribute)) for attribute in value.__slots__)
        return total

    return size(root)


def bloom_metrics(contains, truth, negatives):
    false_positives = sum(contains(x) for x in negatives)
    false_negatives = sum(not contains(x) for x in truth)
    return {
        "fp": false_positives, "fn": false_negatives,
        "fpr": false_positives / len(negatives), "fnr": false_negatives / len(truth),
    }


def cms_metrics(estimate, truth, negatives):
    guesses = {item: estimate(item) for item in truth}
    errors = {item: guesses[item] - truth[item] for item in truth}

    def relative_error(items):
        return statistics.mean(errors[item] / truth[item] for item in items)

    expected_top = set(sorted(truth, key=lambda x: (-truth[x], x))[:20])
    estimated_top = set(sorted(guesses, key=lambda x: (-guesses[x], x))[:20])
    return {
        "mae": statistics.mean(abs(error) for error in errors.values()),
        "mre": relative_error(truth),
        "rare_mre": relative_error(item for item in truth if truth[item] <= 10),
        "popular_mre": relative_error(item for item in truth if truth[item] >= 1000),
        "max_overcount": max(errors.values()),
        "exact_rate": sum(error == 0 for error in errors.values()) / len(truth),
        "underestimates": sum(error < 0 for error in errors.values()),
        "top20_hits": len(estimated_top & expected_top),
        "absent_nonzero": sum(estimate(x) > 0 for x in negatives) / len(negatives),
    }


def consume(archive, kind, params):
    if kind == "set":
        return set(movie_ids(archive))
    if kind == "dict":
        return Counter(movie_ids(archive))
    sketch = BloomFilter(*params) if kind == "bloom" else CountMinSketch(*params)
    add = sketch.add
    for movie_id in movie_ids(archive):
        add(movie_id)
    return sketch


def evaluate(kind, params, sketch, truth, negatives):
    measurements = {"retained_bytes": footprint(sketch), "payload_bytes": None}
    if kind == "bloom":
        m, k = params
        measurements.update(
            m=m, k=k, payload_bytes=len(sketch.bits),
            occupancy=sum(byte.bit_count() for byte in sketch.bits) / m,
            theoretical_fpr=(1 - math.exp(-k * len(truth) / m)) ** k,
            **bloom_metrics(sketch.contains, truth, negatives),
        )
        assert measurements["fn"] == 0
    elif kind == "cms":
        width, depth = params
        measurements.update(
            width=width, depth=depth, payload_bytes=len(sketch.counts) * sketch.counts.itemsize,
            **cms_metrics(sketch.estimate, truth, negatives),
        )
        assert measurements["underestimates"] == 0
    elif kind == "set":
        assert len(sketch) == len(truth)
        measurements.update(bloom_metrics(sketch.__contains__, truth, negatives))
    else:
        assert sketch == truth
        measurements.update(cms_metrics(lambda x: sketch.get(x, 0), truth, negatives))
    return measurements


def benchmark(archive, output, repeats):
    if repeats < 1:
        raise ValueError("At least one timing repetition is required.")
    prepare(archive)
    sha256 = verify_archive(archive)
    truth = Counter(movie_ids(archive))
    count = sum(truth.values())
    if count < 100_000:
        raise ValueError(f"Expected at least 100,000 ratings, got {count:,}.")
    domain_max = max(truth) + 20_000
    negatives = random.Random(SEED).sample(
        [item for item in range(1, domain_max + 1) if item not in truth], NEGATIVE_QUERIES,
    )
    configs = [("set", ()), ("dict", ())] + [
        ("bloom", params) for params in BLOOM_PARAMS
    ] + [("cms", params) for params in CMS_PARAMS]
    times = {config: [] for config in configs}
    measured = {}
    rng = random.Random(SEED)
    for trial in range(repeats):
        order = configs.copy()
        rng.shuffle(order)
        for config in order:
            kind, params = config
            start = perf_counter()
            sketch = consume(archive, kind, params)
            elapsed = perf_counter() - start
            times[config].append(elapsed)
            if trial == 0:
                measured[config] = evaluate(kind, params, sketch, truth, negatives)
            del sketch
            print(f"{trial + 1}/{repeats}  {kind:5} {str(params):14} {elapsed:6.3f} s", flush=True)

    rows = []
    for config in configs:
        timings = times[config]
        seconds = statistics.median(timings)
        rows.append(dict(kind=config[0], **measured[config], seconds=seconds,
                         min_seconds=min(timings), max_seconds=max(timings),
                         records_per_second=count / seconds, trials=timings))
    summary = dict(
        records=count, distinct=len(truth), min_id=min(truth), max_id=max(truth),
        rare_count=sum(n <= 10 for n in truth.values()),
        popular_count=sum(n >= 1000 for n in truth.values()),
        most_common=truth.most_common(5), negative_queries=len(negatives),
        negative_domain_max=domain_max, repeats=repeats, seed=SEED, sha256=sha256,
        python=platform.python_version(), implementation=platform.python_implementation(),
        system=platform.system(), release=platform.release(), arch=platform.machine(),
        cpu=os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine(),
        counter_bytes=array("I").itemsize,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps({"environment": summary, "results": rows}, indent=2), encoding="utf-8",
    )
    fields = list(dict.fromkeys(key for item in rows for key in item if key != "trials"))
    with (output / "benchmark.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{count:,} ratings; {len(truth):,} distinct movies. Results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.archive, args.output, args.repeats)
