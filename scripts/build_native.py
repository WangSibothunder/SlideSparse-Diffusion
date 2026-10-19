"""Explicit offline build of the bundled SDK; never run by package installation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutlass', type=Path, required=True,
                        help='CUTLASS checkout at reference/cutlass_pin.json revision')
    args = parser.parse_args()
    from sdacc import w8a8_slidesparse as sdk
    root = Path(sdk.__file__).resolve().parent
    sdk.build_native(root / 'native_sm80', args.cutlass.resolve(), fused_projection=True)
    print('Built native SDK in', root / 'native_sm80')


if __name__ == '__main__':
    main()
