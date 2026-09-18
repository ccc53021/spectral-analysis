"""Nested 6+14 projection of the complete unrestricted Ascon 2+4 model.

The selected candidate order is fixed. The computation extends the greedy rank
from eight to fourteen masks and evaluates the C-generated endpoint graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np

import utils
from model import Graph, Prefix
from utils import bits, dump, fraction_record, fwht, mask_expression, rref_basis, sha, sha_range, span

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'
B_LABELS = ('K0[0]','K1[0]','K0[45]','K1[45]','Q0','Q45')


def branch_basis():
    result = [utils.words_to_int(utils.column_words(1 << (4-r), c))
              for c in (0,45) for r in (1,2)]
    result += [utils.words_to_int(utils.column_words(3,c)) for c in (0,45)]
    assert len(rref_basis(result)) == 6
    return result


def choose_basis(selection):
    rows = selection['candidates']
    raw_old = [utils.words_to_int(utils.parse_words(x)) for x in selection['old_basis']]
    b6, raw14, selected_rows = branch_basis(), [], []
    for row in rows:
        if not float(row['coefficient']):
            continue
        mask = utils.words_to_int(utils.parse_words(row['mask_words']))
        if len(rref_basis(b6+raw14+[mask])) > len(rref_basis(b6+raw14)):
            raw14.append(mask); selected_rows.append(row)
            if len(raw14) == 14:
                break
    assert raw14[:8] == raw_old
    readable8 = [utils.words_to_int(utils.parse_words(x)) for x in selection['readable_basis']]
    assert set(span(readable8)) == set(span(raw_old))
    final14 = readable8 + raw14[8:]
    assert len(rref_basis(b6+final14)) == 20
    return raw14, final14, selected_rows


def dyadic_key(value, exponent):
    value = int(value)
    if not value:
        return (0,0)
    magnitude = abs(value)
    shift = (magnitude & -magnitude).bit_length()-1
    return (value >> shift, exponent-shift)


def key_fraction(key):
    numerator, exponent = key
    return Fraction(numerator, 1 << exponent)


def key_record(key, **extra):
    row = fraction_record(key_fraction(key))
    row.update(denominator_exponent=key[1], **extra)
    return row


def integer_fwht(values):
    values = list(map(int,values)); h = 1
    while h < len(values):
        for start in range(0,len(values),2*h):
            for j in range(start,start+h):
                a,b = values[j],values[j+h]
                values[j],values[j+h] = a+b,a-b
        h *= 2
    return values


def to_common_integers(values):
    exponents = []
    for value in values:
        denominator = value.denominator
        assert denominator & (denominator-1) == 0
        exponents.append(denominator.bit_length()-1)
    exponent = max(exponents,default=0)
    integers = [value.numerator << (exponent-e) for value,e in zip(values,exponents)]
    return integers,exponent


def bindex(text):
    assert len(text)==6
    return sum(int(x)<<j for j,x in enumerate(text))


def load_graph(branch, manifest):
    metadata = manifest[branch]
    path = DATA / 'graphs.bin'
    assert sha_range(path,metadata['offset'],metadata['size']) == metadata['sha256']
    return Graph(path,metadata['offset'],metadata['size']), metadata


def compute(out=OUT):
    started = time.perf_counter()
    selection = json.loads((DATA/'selection.json').read_text())
    profiles = json.loads((DATA/'profiles.json').read_text())['rows']
    graphs = json.loads((DATA/'graphs.json').read_text())
    assert len(profiles) == len(graphs) == 64
    graph_path = DATA/'graphs.bin'
    if not graph_path.exists():
        raise FileNotFoundError("graphs.bin is missing; run 'python generate_graphs.py' first")
    expected_size = sum(metadata['size'] for metadata in graphs.values())
    if graph_path.stat().st_size != expected_size:
        raise RuntimeError("graphs.bin has an unexpected size; rerun 'python generate_graphs.py'")
    out.mkdir(parents=True,exist_ok=False)
    raw14,basis,selected_rows = choose_basis(selection)
    request = dict(mode='unrestricted',split='2+4',
        selection='continue the frozen 4096-candidate complete-score order to fourteen independent masks modulo B6',
        old_extra_dimension=8,new_extra_dimension=14,total_dimension=20,class_count=1<<20,
        branch_count=64,classes_per_branch=1<<14,free_input_bits_per_class=236,
        class_probability=fraction_record(Fraction(1,1<<20)),
        raw_selected_basis=[dict(index=i,expression=mask_expression(m),mask_words=utils.words_hex(utils.int_to_words(m)),
                                 archived_power=selected_rows[i]['power']) for i,m in enumerate(raw14)],
        final_basis=[dict(index=i,expression=mask_expression(m),mask_words=utils.words_hex(utils.int_to_words(m)),
                          source='old readable U8' if i<8 else 'next frozen greedy mask') for i,m in enumerate(basis)],
        combined_rank=len(rref_basis(branch_basis()+basis)),nested_U8=True,
        scalar_graphs='complete float64 four-round RB endpoint functions generated by route_c_scalar_evadd.c',
        arithmetic='exact dyadic connection and integer FWHT relative to stored float64 graph edges',
        input_hashes={name:sha(DATA/name) for name in ('selection.json','profiles.json','graphs.json')},
        script_sha256=sha(Path(__file__)))
    dump(out/'request.json',request)
    n=1<<14
    class_keys_by_branch={}
    branch_coefficients={}
    branch_records=[]
    for position,profile in enumerate(profiles):
        t0=time.perf_counter();name=profile['branch_bits'];graph,meta=load_graph(name,graphs)
        prefix=Prefix(profile);uniform,proof=prefix.plan(0);assert uniform==[(0,Fraction(1))]
        masks=span(basis)
        if graph.root==0:
            plans=[[] for _ in masks];diagnostics=[dict(reason='pointwise zero scalar graph') for _ in masks]
        else:
            planned=[prefix.plan(mask) for mask in masks]
            plans,diagnostics=zip(*planned)
        vs=sorted({v for terms in plans for v,_ in terms})
        exact={v:graph.exact_query(v) for v in vs}
        coefficients=[sum((exact[v]*g for v,g in terms),Fraction()) for terms in plans]
        coefficient_integers,exponent=to_common_integers(coefficients)
        class_integers=integer_fwht(coefficient_integers)
        assert sum(class_integers)==n*coefficient_integers[0]
        keys=[dyadic_key(x,exponent) for x in class_integers]
        class_keys_by_branch[bindex(name)]=keys
        sparse={i:value for i,value in enumerate(coefficients) if value}
        branch_coefficients[bindex(name)]=sparse
        counts=Counter(keys)
        local_values=sorted(counts,key=key_fraction)
        local_id={key:i for i,key in enumerate(local_values)}
        ids=np.array([local_id[key] for key in keys],dtype=np.uint16 if len(local_values)<65536 else np.uint32)
        np.savez_compressed(out/f'class_ids_{name}.npz',class_ids=ids)
        record=dict(complete=True,branch_bits=name,branch_index=bindex(name),second_label_dimension=profile['second_label_dimension'],
            second_candidates=profile['second_candidates'],graph_sha256=meta['sha256'],graph_reachable_nodes=len(graph.levels),
            exact_uniform_T_certificate=proof,extra_dimension=14,class_count=n,
            kernel_V_queries=len(vs),kernel_V_values=vs,nonempty_input_characters=sum(bool(x) for x in plans),
            total_kernel_terms=sum(map(len,plans)),max_kernel_support_bound=max((d.get('support_bound',0) for d in diagnostics),default=0),
            coefficient_nonzero=len(sparse),coefficient_common_denominator_exponent=exponent,
            mean=fraction_record(coefficients[0]),minimum=fraction_record(key_fraction(local_values[0])),
            maximum=fraction_record(key_fraction(local_values[-1])),distinct_values=len(local_values),
            values=[key_record(key,id=i,class_count=counts[key]) for i,key in enumerate(local_values)],
            sparse_coefficients=[dict(index=i,**fraction_record(value)) for i,value in sparse.items()],
            elapsed_seconds=time.perf_counter()-t0)
        dump(out/f'branch_{name}.json',record);branch_records.append(record)
        print('branch',position+1,'/64',name,'r',profile['second_label_dimension'],'V',len(vs),
              'coef',len(sparse),'values',len(local_values),'range',record['minimum']['power'],record['maximum']['power'],
              'seconds',record['elapsed_seconds'],flush=True)
    # Global class table: value dictionary plus a compact 64 x 2^14 ID array.
    all_keys=sorted(set(key for keys in class_keys_by_branch.values() for key in keys),key=key_fraction)
    key_id={key:i for i,key in enumerate(all_keys)}
    dtype=np.uint16 if len(all_keys)<65536 else np.uint32
    class_ids=np.empty((64,n),dtype=dtype)
    distribution=Counter()
    for b,keys in class_keys_by_branch.items():
        class_ids[b]=[key_id[key] for key in keys];distribution.update(keys)
    np.savez_compressed(out/'classes_20d.npz',class_value_ids=class_ids)
    dump(out/'class_value_table.json',dict(values=[key_record(key,id=i,class_count=distribution[key],
        probability=fraction_record(Fraction(distribution[key],1<<20))) for i,key in enumerate(all_keys)]))
    # Sparse full 20D spectrum, computed from per-branch coefficients without a million-Fraction FWHT.
    spectrum_keys=[];spectrum_indices=[]
    for s in sorted(set(i for row in branch_coefficients.values() for i in row)):
        transformed=fwht([branch_coefficients[b].get(s,Fraction()) for b in range(64)])
        for k,value in enumerate(transformed):
            value/=64
            if value:
                spectrum_indices.append(k+(s<<6));spectrum_keys.append((value.numerator,value.denominator.bit_length()-1))
    spectrum_values=sorted(set(spectrum_keys),key=key_fraction);spectrum_id={key:i for i,key in enumerate(spectrum_values)}
    np.savez_compressed(out/'spectrum_20d.npz',indices=np.array(spectrum_indices,dtype=np.uint32),
        value_ids=np.array([spectrum_id[key] for key in spectrum_keys],dtype=np.uint16 if len(spectrum_values)<65536 else np.uint32))
    dump(out/'spectrum_value_table.json',dict(values=[key_record(key,id=i) for i,key in enumerate(spectrum_values)]))
    mean=sum((key_fraction(key)*count for key,count in distribution.items()),Fraction())/(1<<20)
    minimum,maximum=all_keys[0],all_keys[-1]
    min_count,max_count=distribution[minimum],distribution[maximum]
    def example(key):
        target=key_id[key];where=np.argwhere(class_ids==target)[0];b,u=map(int,where)
        return dict(global_index=b+(u<<6),branch_index=b,branch_bits=bits(b,6),extra_index=u,extra_bits=bits(u,14))
    summary=dict(complete=True,dimension=20,branch_dimension=6,extra_dimension=14,class_count=1<<20,
        free_input_bits_per_class=236,class_probability=fraction_record(Fraction(1,1<<20)),
        selected_basis=request['final_basis'],raw_selected_basis=request['raw_selected_basis'],combined_rank=20,
        mean=fraction_record(mean),minimum=key_record(minimum,class_count=min_count,example=example(minimum)),
        maximum=key_record(maximum,class_count=max_count,example=example(maximum)),
        zero_class_count=distribution.get((0,0),0),distinct_values=len(all_keys),
        coefficient_nonzero=len(spectrum_indices),spectrum_distinct_values=len(spectrum_values),
        storage='classes_20d.npz maps [branch_index,extra_index] to class_value_table.json; spectrum analogous',
        branch_records=[{k:v for k,v in row.items() if k not in ('values','sparse_coefficients')} for row in branch_records],
        model_boundary='exact selected-space projection relative to stored float64 RB4 suffix graphs',
        elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selected_basis','raw_selected_basis','branch_records')},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description='Ascon DL.8 prefix second-order computation')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Preserving existing output directory: {args.output}')
    compute(args.output)


if __name__=='__main__':
    main()

