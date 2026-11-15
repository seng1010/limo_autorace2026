import sys
sys.path.insert(0, '.'); import sim5, e9, e10
tag, path = sys.argv[1].split('=')
for title, smin, smax in (('줄 1.2~1.4', 1.2, 1.4), ('줄 1.0~1.2', 1.0, 1.2), ('줄 0.85~1.0', 0.85, 1.0)):
    print(title); sys.stdout.flush(); e9.evaluate(tag, path, e10.rows7(40, 31, smin, smax)); sys.stdout.flush()
