"""Write the RQ3 and RQ4 model inputs using the paper experiment generators."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qamc.experiments.rq3 import RQ3_BENCHMARKS, RQ3_DEFAULT_SIZES
from qamc.experiments.rq4 import RQ4_BENCHMARKS, RQ4_DEFAULT_SIZES


def model_sources(rq3_sizes, rq4_sizes):
    for benchmark in RQ3_BENCHMARKS.values():
        for n in rq3_sizes:
            for style in ("abstract", "full"):
                yield f"rq3/{benchmark.id}/{style}/{n}.qam", benchmark.model_source(n, style)
    for benchmark in RQ4_BENCHMARKS.values():
        for n in rq4_sizes:
            yield f"rq4/{benchmark.id}/{n}.qam", benchmark.model_source(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/scaling_models.zip"),
                        help="Path to the generated ZIP archive.")
    parser.add_argument("--rq3-sizes", type=int, nargs="+", default=RQ3_DEFAULT_SIZES,
                        help="Register sizes for the four paired families (default: 5 through 20).")
    parser.add_argument("--rq4-sizes", type=int, nargs="+", default=RQ4_DEFAULT_SIZES,
                        help="Register sizes for the five scaling families, in multiples of 10 "
                             "(default: 50 100 200 300 400 500).")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as working:
        archive = Path(working) / "models.zip"
        with ZipFile(archive, "w") as out:
            for name, source in model_sources(sorted(set(args.rq3_sizes)), sorted(set(args.rq4_sizes))):
                entry = ZipInfo(name)  # Fixed member timestamp keeps archives reproducible.
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                out.writestr(entry, source.encode("utf-8"), compress_type=ZIP_DEFLATED)
        archive.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
