from argparse import ArgumentParser
from pathlib import Path

from frt.audio.sample_library import SampleLibrary
from frt.dataset.synthetic import RiffGenerationConfig, write_dataset


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate synthetic guitar riff data.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--technique-index", type=int, default=0)
    parser.add_argument("--min-fret", type=int, default=0)
    parser.add_argument("--max-fret", type=int, default=24)
    parser.add_argument("--min-volume", type=float, default=0.5)
    parser.add_argument("--max-volume", type=float, default=1.0)
    parser.add_argument("--raw-root", type=Path, default=Path("samples/raw"))
    parser.add_argument("--cache-path", type=Path, default=Path("samples.npy"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RiffGenerationConfig(
        duration_seconds=args.duration_seconds,
        min_events=args.min_events,
        max_events=args.max_events,
        technique_index=args.technique_index,
        min_fret=args.min_fret,
        max_fret=args.max_fret,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
    )
    library = SampleLibrary.load(raw_root=args.raw_root, cache_path=args.cache_path)
    entries = write_dataset(
        library=library,
        output_dir=args.out,
        count=args.count,
        config=config,
        seed=args.seed,
    )
    print(f"Wrote {len(entries)} examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
