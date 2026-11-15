import sys
sys.path.insert(0, '.'); import sim5, e9, e10
tag, path = sys.argv[1].split('=')
print('7대 4줄, 줄 간격 1.0~1.2 m, 40개'); e9.evaluate(tag, path, e10.rows7(40, 31, 1.0, 1.2), show=2)
