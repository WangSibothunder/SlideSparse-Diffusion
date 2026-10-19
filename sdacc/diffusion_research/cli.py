"""One entry point. Metadata commands neither import torch nor start experiments."""
import argparse,json
from dataclasses import asdict
from .registry import MODELS,definition

def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m sdacc.diffusion_research')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('models',help='list model adapters and planned extensions')
    result=sub.add_parser('results',help='read preserved results; never rerun')
    result.add_argument('--model');result.add_argument('--index')
    run=sub.add_parser('run',help='explicit model execution through the shared runtime')
    run.add_argument('--model',required=True);run.add_argument('--config',required=True)
    run.add_argument('--stage',choices=['infer','calibrate','benchmark','profile','decode','score','study'],required=True)
    run.add_argument('--gpu',type=int,help='physical GPU index, or inherit an already selected single GPU');run.add_argument('--output',required=True);run.add_argument('--policy');run.add_argument('--affine',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='models':value=[asdict(x) for x in MODELS.values()]
    elif args.command=='results':
        from .results import catalog
        value=catalog(args.model,args.index)
    else:
        if args.stage=='calibrate' and args.affine:parser.error('calibration must use the uncorrected model')
        if definition(args.model).status!='implemented':parser.error(definition(args.model).notes)
        from .config import load
        from .runner import execute
        execute(load(args.config,args.model),args)
        return
    print(json.dumps(value,indent=2,ensure_ascii=False))
