"""Tables for article v10 (SIAM format): a reduced version of the Pi table for the main text.
The full table is ../../experiments/tables/table_index.tex (supplement). Rows are copied unchanged.
Input : ../../experiments/tables/table_index.tex
Output: ../tables/table_v10_index_main.tex
"""
keep = [r'$\delta=10^{-2}$', r'$\delta=10^{-4}$', r'$\delta=10^{-6}$', r'$\delta=10^{-1}$', r'$\delta=10^{-3}$',
        r'$\varepsilon=10^{-2}$', r'$\varepsilon=10^{-8}$', r'$x_0=10^{-1}$', r'$x_0=10^{-7}$',
        r'$I_0=10^{-2}$', r'$I_0=10^{-8}$']
rows = open('../../experiments/tables/table_index.tex').read().strip().splitlines()
out = []
for r in rows:
    label = r.split('&')[0]
    q3 = 'y^3' in label
    if q3 and (r'$\delta=10^{-1}$' in label or r'$\delta=10^{-3}$' in label):
        out.append(r)
    elif not q3 and 'y^2' in label and any(k in label for k in keep[:3]):
        out.append(r)
    elif not q3 and 'y^2' not in label and any(k in label for k in keep[5:]):
        out.append(r)
open('../tables/table_v10_index_main.tex', 'w').write('\n'.join(out) + '\n')
print(len(rows), '->', len(out)); print('\n'.join(o[:50] for o in out))
