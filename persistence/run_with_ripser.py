import subprocess
import shlex
from pathlib import Path


def main(flags):
    flags.output.mkdir(exist_ok=True)
    out_path = lambda f: flags.output / f'{f.stem}.ph.txt'
    files = list(filter(lambda f: not out_path(f).exists(), flags.files))
    done, cnt = 0, len(files)
    files.reverse()

    while len(files) > 0:
        ps = []
        for _ in range(flags.threads):
            f = files[-1]; files.pop()
            bcomplex = f"./vcomplex {f} {flags.mode} "
            ripser = f"./ripser/ripser --dim {flags.dim}"
            of = out_path(f).open('w')
            p1 = subprocess.Popen(shlex.split(bcomplex), stdout=subprocess.PIPE)
            p2 = subprocess.Popen(shlex.split(ripser), stdin=p1.stdout, stdout=of)
            ps.append((p2, of))
            if len(files) == 0:
                break
        done += len(ps)
        print(f'Processing {done}/{cnt}', end='\r')
        for process, of in ps:
            process.wait()
            of.close()
        
    print(f'Processed {cnt}' + ' '*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', '-j', type=int, default=2)
    parser.add_argument('files', type=Path, nargs="+")
    parser.add_argument('--mode', '-m', type=str)
    parser.add_argument('--verbose', '-v', action="store_true")
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('out'))
    args = parser.parse_args()

    main(args)
