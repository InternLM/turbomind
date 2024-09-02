import json

import fire
import prettytable


def get_tag(state, tag):
    for x in state['summaries']:
        if x['tag'] == tag:
            return float(x['data'][0]['value'])


def main(path_a: str, path_b: str):

    with open(path_a, 'r') as f:
        a = json.load(f)
    with open(path_b, 'r') as f:
        b = json.load(f)
    states_a = a['benchmarks'][0]['states']
    states_b = b['benchmarks'][0]['states']
    by_idx = {}
    by_idx_4 = {}
    by_bs = {}
    table = prettytable.PrettyTable(['idx', 'bs', 'tp', 'flops', 'diff'])
    for u in states_a:
        for v in states_b:
            if u['name'] == v['name']:
                break
        assert u['name'] == v['name']
        tag = 'nv/cold/bw/item_rate'
        flops_a = get_tag(u, tag)
        flops_b = get_tag(v, tag)
        ratio = flops_a / flops_b
        flops = round(flops_a / 1e12, 3)
        diff = round((ratio - 1.0) * 100, 3)
        idx, bs, tp = u['name'].split(' ')[1:]
        table.add_row([idx, bs, tp, flops, diff])
        x = by_idx.get(idx, [0, 0, 0])
        by_idx[idx] = x[0] + 1, x[1] + flops, x[2] + diff
        x = by_bs.get(bs, [0, 0, 0])
        by_bs[bs] = x[0] + 1, x[1] + flops, x[2] + diff
        idx_4 = int(idx[4:]) % 4
        x = by_idx_4.get(idx_4, [0, 0, 0])
        by_idx_4[idx_4] = x[0] + 1, x[1] + flops, x[2] + diff

    by_idx_tab = prettytable.PrettyTable(['idx', 'flops', 'diff'])
    for k, (cnt, flops, diff) in by_idx.items():
        flops = round(flops / cnt, 3)
        diff = round(diff / cnt, 3)
        by_idx_tab.add_row([k, flops, diff])

    by_bs_tab = prettytable.PrettyTable(['idx', 'flops', 'diff'])
    for k, (cnt, flops, diff) in by_bs.items():
        flops = round(flops / cnt, 3)
        diff = round(diff / cnt, 3)
        by_bs_tab.add_row([k, flops, diff])

    by_idx_4_tab = prettytable.PrettyTable(['idx_4', 'flops', 'diff'])
    for k, (cnt, flops, diff) in by_idx_4.items():
        flops = round(flops / cnt, 3)
        diff = round(diff / cnt, 3)
        by_idx_4_tab.add_row([k, flops, diff])

    print(table)

    print(by_idx_tab)
    print(by_idx_4_tab)
    print(by_bs_tab)


if __name__ == '__main__':
    fire.Fire(main)
