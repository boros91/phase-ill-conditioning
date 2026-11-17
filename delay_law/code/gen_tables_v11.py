"""Tables for article v11: the MATLAB method-constant table with a reader-facing column header
(rows copied unchanged from ../../experiments/tables/table_matlab_v3.tex).
Output: ../tables/table_v11_matlab.tex
"""
s = open('../../experiments/tables/table_matlab_v3.tex').read()
assert s.count('flame, runs of Paper A') == 1
open('../tables/table_v11_matlab.tex', 'w').write(s.replace('flame, runs of Paper A', 'flame, companion-paper runs'))
print('ok')
